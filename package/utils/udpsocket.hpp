#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <optional>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <boost/asio.hpp>

// (The UdpSocketConfig struct and type aliases are unchanged)
struct UdpSocketConfig {
    std::string host;
    uint16_t port;
    std::optional<std::string> localInterfaceIp;
    uint32_t bufferSize = 65535;
    std::chrono::milliseconds resolveTimeout{100000};
    bool enableBroadcast = false;
    std::optional<boost::asio::ip::address> multicastGroup;
    std::optional<std::chrono::milliseconds> receiveTimeout;
    std::optional<int> ttl;
    bool reuseAddress = true;
};

using DataBuffer = std::vector<uint8_t>;
using UdpSocketPtr = std::shared_ptr<class UdpSocket>;
using DataCallback = std::function<void(std::unique_ptr<DataBuffer>)>;
using ErrorCallback = std::function<void(const boost::system::error_code&)>;


class UdpSocket : public std::enable_shared_from_this<UdpSocket> {
public:
    UdpSocket(
        boost::asio::io_context& context,
        DataCallback dataCallback,
        ErrorCallback errorCallback,
        const UdpSocketConfig& config)
        : socket_(context),
          resolver_(context),
          timeoutTimer_(context),
          buffer_(config.bufferSize),
          dataCallback_(std::move(dataCallback)),
          errorCallback_(std::move(errorCallback)),
          config_(config) {
        if (config.bufferSize == 0) {
            throw std::invalid_argument("Buffer size must be non-zero");
        }
    }

    static UdpSocketPtr create(
        boost::asio::io_context& context,
        const UdpSocketConfig& config,
        DataCallback dataCallback,
        ErrorCallback errorCallback) {
        if (config.host.empty()) {
            throw std::invalid_argument("Host cannot be empty");
        }
        if (config.port == 0) {
            throw std::invalid_argument("Port must be non-zero");
        }
        if (config.multicastGroup && !config.multicastGroup->is_multicast()) {
            throw std::invalid_argument("Invalid multicast group address");
        }
        if (config.multicastGroup && !config.localInterfaceIp) {
            throw std::invalid_argument("localInterfaceIp must be set for multicast");
        }
        auto socket = std::make_shared<UdpSocket>(context, std::move(dataCallback), std::move(errorCallback), config);
        socket->start();
        return socket;
    }

    ~UdpSocket() { stop(); }

    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    UdpSocket(UdpSocket&&) = delete;
    UdpSocket& operator=(UdpSocket&&) = delete;

    void stop() {
        boost::system::error_code ec;
        timeoutTimer_.cancel(ec);
        resolver_.cancel();
        if (socket_.is_open()) {
            socket_.cancel(ec);
            socket_.close(ec);
        }
    }

private:
    void start() {
        if (config_.multicastGroup) {
            setupMulticastSocket();
        } else {
            startResolve();
        }
    }

    void startResolve() {
        resolver_.async_resolve(
            boost::asio::ip::udp::v4(), config_.host, std::to_string(config_.port),
            [self = shared_from_this()](const boost::system::error_code& ec, const auto& endpoints) {
                self->handleResolve(ec, endpoints);
            });
        timeoutTimer_.expires_after(config_.resolveTimeout);
        timeoutTimer_.async_wait([self = shared_from_this()](const boost::system::error_code& ec) {
            if (!ec) {
                self->resolver_.cancel();
                if (self->errorCallback_) self->errorCallback_(boost::asio::error::timed_out);
            }
        });
    }

    void handleResolve(const boost::system::error_code& ec, const boost::asio::ip::udp::resolver::results_type& endpoints) {
        timeoutTimer_.cancel();
        if (ec == boost::asio::error::operation_aborted) return;
        if (ec) {
            if (errorCallback_) errorCallback_(ec);
            return;
        }
        setupSocketAndBind(*endpoints.begin());
    }
    
    void setupMulticastSocket() {
        boost::system::error_code ec;

        boost::asio::ip::udp::endpoint listenEndpoint(boost::asio::ip::udp::v4(), config_.port);
        
        socket_.open(listenEndpoint.protocol(), ec);
        if (ec) { if (errorCallback_) errorCallback_(ec); return; }

        if (config_.reuseAddress) {
            socket_.set_option(boost::asio::socket_base::reuse_address(true), ec);
            if (ec) { if (errorCallback_) errorCallback_(ec); return; }
        }
        
        socket_.bind(listenEndpoint, ec);
        if (ec) {
            std::cerr << "Multicast bind error on port " << config_.port << ": " << ec.message() << std::endl;
            if (errorCallback_) errorCallback_(ec); 
            return;
        }

        boost::asio::ip::address localInterfaceAddress = boost::asio::ip::make_address(*config_.localInterfaceIp, ec);
        if (ec) {
            std::cerr << "Invalid local interface IP '" << *config_.localInterfaceIp << "': " << ec.message() << std::endl;
            if (errorCallback_) errorCallback_(ec);
            return;
        }

        // ###############################################################
        // ##                  MODIFIED SECTION START                   ##
        // ###############################################################

        // Ensure both addresses are IPv4 before attempting to join the group.
        if (!config_.multicastGroup->is_v4() || !localInterfaceAddress.is_v4()) {
            std::cerr << "Multicast group and local interface addresses must be IPv4." << std::endl;
            // Create a custom error code if desired, or use an existing one.
            if (errorCallback_) errorCallback_(boost::asio::error::address_family_not_supported);
            return;
        }

        // Explicitly convert the generic `address` types to `address_v4` using .to_v4()
        socket_.set_option(
            boost::asio::ip::multicast::join_group(
                config_.multicastGroup->to_v4(),
                localInterfaceAddress.to_v4()
            ), 
            ec
        );

        // ###############################################################
        // ##                   MODIFIED SECTION END                    ##
        // ###############################################################

        if (ec) {
            std::cerr << "Join multicast group error: " << ec.message() << std::endl;
            if (errorCallback_) errorCallback_(ec);
            return;
        }
        
        startReceive();
    }

    void setupSocketAndBind(const boost::asio::ip::udp::endpoint& endpoint) {
        boost::system::error_code ec;
        socket_.open(endpoint.protocol(), ec);
        if (ec) { if (errorCallback_) errorCallback_(ec); return; }

        if (config_.reuseAddress) {
            socket_.set_option(boost::asio::socket_base::reuse_address(true), ec);
            if (ec) { if (errorCallback_) errorCallback_(ec); return; }
        }

        socket_.bind(endpoint, ec);
        if (ec) { if (errorCallback_) errorCallback_(ec); return; }

        if (config_.enableBroadcast) {
            socket_.set_option(boost::asio::socket_base::broadcast(true), ec);
            if (ec) { if (errorCallback_) errorCallback_(ec); return; }
        }
        
        if (config_.ttl) {
            socket_.set_option(boost::asio::ip::multicast::hops(*config_.ttl), ec);
            if (ec) { if (errorCallback_) errorCallback_(ec); }
        }
        
        startReceive();
    }

    void startReceive() {
        socket_.async_receive_from(
            boost::asio::buffer(buffer_), senderEndpoint_,
            [self = shared_from_this()](const boost::system::error_code& ec, std::size_t bytes) {
                self->handleReceive(ec, bytes);
            });

        if (config_.receiveTimeout) {
            timeoutTimer_.expires_after(*config_.receiveTimeout);
            timeoutTimer_.async_wait([self = shared_from_this()](const boost::system::error_code& ec) {
                if (!ec) {
                    self->socket_.cancel();
                    if (self->errorCallback_) self->errorCallback_(boost::asio::error::timed_out);
                }
            });
        }
    }

    void handleReceive(const boost::system::error_code& ec, std::size_t bytesReceived) {
        if (config_.receiveTimeout) {
            timeoutTimer_.cancel();
        }

        if (!ec) {
            if (bytesReceived > 0) {
                auto packet_ptr = std::make_unique<DataBuffer>(buffer_.begin(), buffer_.begin() + bytesReceived);
                if (dataCallback_) {
                    dataCallback_(std::move(packet_ptr));
                }
                // DataBuffer data(buffer_.begin(), buffer_.begin() + bytesReceived);
                // if (dataCallback_) dataCallback_(data);
            }
        } else if (ec == boost::asio::error::operation_aborted) {
            // Expected on timeout or stop(). Do nothing.
        } else {
            if (errorCallback_) errorCallback_(ec);
        }

        if (socket_.is_open()) {
            startReceive();
        }
    }

    boost::asio::ip::udp::socket socket_;
    boost::asio::ip::udp::resolver resolver_;
    boost::asio::ip::udp::endpoint senderEndpoint_;
    boost::asio::steady_timer timeoutTimer_;
    std::vector<uint8_t> buffer_;
    DataCallback dataCallback_;
    ErrorCallback errorCallback_;
    UdpSocketConfig config_;
};