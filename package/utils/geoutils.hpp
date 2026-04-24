#pragma once

#include <eigenreference.hpp>
#include <cmath>

namespace GeoUtils {

    // WGS84 Ellipsoid Constants
    constexpr double WGS84_A = 6378137.0;             // Semi-major axis (meters)
    constexpr double WGS84_F = 1.0 / 298.257223563;   // Flattening
    constexpr double WGS84_E2 = WGS84_F * (2.0 - WGS84_F); // Square of eccentricity

    inline double SymmetricalAngle(double x) {
        constexpr double PI = M_PI;
        constexpr double TWO_PI = 2.0 * M_PI;
        double y = std::remainder(x, TWO_PI);
        if (y == PI) {y = -PI;}
        return y;
    }

    /**
     * Converts Geodetic (Latitude, Longitude, Altitude) to ECEF (X, Y, Z)
     * * @param lat Latitude in Radians
     * @param lon Longitude in Radians
     * @param alt Altitude in Meters
     * @return Eigen::Vector3d (x, y, z) in ECEF
     */
    inline Eigen::Vector3d LLAtoECEF(double lat, double lon, double alt) {
        double sin_lat = std::sin(lat);
        double cos_lat = std::cos(lat);
        double sin_lon = std::sin(lon);
        double cos_lon = std::cos(lon);

        // Prime Vertical Radius of Curvature N(phi)
        double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);

        double x = (N + alt) * cos_lat * cos_lon;
        double y = (N + alt) * cos_lat * sin_lon;
        double z = (N * (1.0 - WGS84_E2) + alt) * sin_lat;

        return Eigen::Vector3d(x, y, z);
    }

    /**
     * Computes the Rotation Matrix from ECEF to NED frame 
     * centered at a specific reference latitude and longitude.
     */
    inline Eigen::Matrix3d nRe(double ref_lat, double ref_lon) {
        double s_lat = std::sin(ref_lat);
        double c_lat = std::cos(ref_lat);
        double s_lon = std::sin(ref_lon);
        double c_lon = std::cos(ref_lon);

        Eigen::Matrix3d R;
        // Row 1: North (Tangent to meridian, pointing North)
        R(0, 0) = -s_lat * c_lon;
        R(0, 1) = -s_lat * s_lon;
        R(0, 2) =  c_lat;

        // Row 2: East (Tangent to parallel, pointing East)
        R(1, 0) = -s_lon;
        R(1, 1) =  c_lon;
        R(1, 2) =  0.0;

        // Row 3: Down (Normal to ellipsoid, pointing Inward)
        R(2, 0) = -c_lat * c_lon;
        R(2, 1) = -c_lat * s_lon;
        R(2, 2) = -s_lat;

        return R;
    }

    /**
     * Exact conversion from LLA to NED.
     * * @param lat Current Latitude (rad)
     * @param lon Current Longitude (rad)
     * @param alt Current Altitude (m)
     * @param rlat Reference Latitude (rad)
     * @param rlon Reference Longitude (rad)
     * @param ralt Reference Altitude (m)
     * @return Eigen::Vector3d (North, East, Down)
     * the translation output should be 
     * the input frame relative to the reference frame 
     * with vector direction from reference frame to input frame 
     * Component	Your Statement	Direction / Mapping
     *  Translation (AtB​)	Position of Input in Reference frame.	Vector points from Reference to Input.
     *  Rotation (ARB​)	Orientation of Input in Reference frame.	Maps (rotates) vectors from Input to Reference.
     */
    inline Eigen::Vector3d LLAtoNED_Exact(double lat, double lon, double alt, 
                                   double rlat, double rlon, double ralt) {
        // 1. Convert both points to ECEF
        Eigen::Vector3d p_ecef = LLAtoECEF(lat, lon, alt);
        Eigen::Vector3d ref_ecef = LLAtoECEF(rlat, rlon, ralt);

        // 2. Vector difference in ECEF
        Eigen::Vector3d delta_ecef = p_ecef - ref_ecef;

        // 3. Rotate difference into NED frame defined at reference point
        Eigen::Matrix3d R_ecef_to_ned = nRe(rlat, rlon);

        return R_ecef_to_ned * delta_ecef;
    }

    // In GeoUtils namespace
    /**
     * Exact conversion from NED to LLA.
     * * @param n Current North (m)
     * @param e Current East (m)
     * @param d Current Down (m)
     * @param rlat Reference Latitude (rad)
     * @param rlon Reference Longitude (rad)
     * @param ralt Reference Altitude (m)
     * @return Eigen::Vector3d (North, East, Down)
     */
    inline Eigen::Vector3d NEDtoLLA_Exact(double n, double e, double d, 
                                        double rlat, double rlon, double ralt) {
        // 1. NED -> ECEF Offset
        Eigen::Matrix3d R_ned_to_ecef = nRe(rlat, rlon).transpose();
        Eigen::Vector3d delta_ecef = R_ned_to_ecef * Eigen::Vector3d(n, e, d);

        // 2. Ref -> ECEF
        Eigen::Vector3d ref_ecef = LLAtoECEF(rlat, rlon, ralt);
        Eigen::Vector3d target_ecef = ref_ecef + delta_ecef;

        // 3. ECEF -> LLA (Standard Closed-Form)
        // Uses WGS84 Constants defined in your file
        double x = target_ecef.x();
        double y = target_ecef.y();
        double z = target_ecef.z();
        
        double lon = std::atan2(y, x);
        double p = std::sqrt(x*x + y*y);
        double theta = std::atan2(z * WGS84_A, p * (WGS84_A * (1.0 - WGS84_F))); // Approx b = A*(1-f)
        
        // Constants from your existing GeoUtils
        double e2 = WGS84_E2;
        double ep2 = (WGS84_A*WGS84_A - (WGS84_A * (1.0 - WGS84_F))*(WGS84_A * (1.0 - WGS84_F))) / ((WGS84_A * (1.0 - WGS84_F))*(WGS84_A * (1.0 - WGS84_F)));

        double lat = std::atan2(z + ep2 * (WGS84_A * (1.0 - WGS84_F)) * std::pow(std::sin(theta), 3), 
                                p - e2 * WGS84_A * std::pow(std::cos(theta), 3));
        
        double sin_lat = std::sin(lat);
        double N = WGS84_A / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
        double alt = p / std::cos(lat) - N;

        return Eigen::Vector3d(lat, lon, alt);
    }

    inline Eigen::Vector3d lla2ned(double lat, double lon, double alt, double rlat, double rlon, double ralt) {
        // Constants according to WGS84
        constexpr double a = 6378137.0;              // Semi-major axis (m)
        constexpr double e2 = 0.00669437999014132;   // Squared eccentricity
        double dphi = lat - rlat;
        double dlam = SymmetricalAngle(lon - rlon);
        double dh = alt - ralt;
        double cp = std::cos(rlat);
        double sp = std::sin(rlat); // Fixed: was sin(originlon)
        double tmp1 = std::sqrt(1 - e2 * sp * sp);
        double tmp3 = tmp1 * tmp1 * tmp1;
        double dlam2 = dlam * dlam;   // Fixed: was dlam.*dlam
        double dphi2 = dphi * dphi;   // Fixed: was dphi.*dphi
        double E = (a / tmp1 + ralt) * cp * dlam -
                (a * (1 - e2) / tmp3 + ralt) * sp * dphi * dlam + // Fixed: was dphi.*dlam
                cp * dlam * dh;                                       // Fixed: was dlam.*dh
        double N = (a * (1 - e2) / tmp3 + ralt) * dphi +
                1.5 * cp * sp * a * e2 * dphi2 +
                sp * sp * dh * dphi +                              // Fixed: was dh.*dphi
                0.5 * sp * cp * (a / tmp1 + ralt) * dlam2;
        double D = -(dh - 0.5 * (a - 1.5 * a * e2 * cp * cp + 0.5 * a * e2 + ralt) * dphi2 -
                    0.5 * cp * cp * (a / tmp1 - ralt) * dlam2);
        return Eigen::Vector3d(N, E, D);
    }
}