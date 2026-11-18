#ifndef CNN_FIXPOINT_H
#define CNN_FIXPOINT_H

#include <cstdint>

namespace cnn {

class FixedPoint {
public:
    // Le format est Q15.16 : 16 bits après la virgule.
    static constexpr int fractional_bits = 16;
    static constexpr std::int32_t scale = 65536;

    FixedPoint();
    FixedPoint(float value);
    FixedPoint(double value);

    static FixedPoint from_raw(std::int32_t raw);

    std::int32_t raw() const;
    float to_float() const;
    double to_double() const;

    FixedPoint operator-() const;
    FixedPoint operator+(FixedPoint other) const;
    FixedPoint operator-(FixedPoint other) const;
    FixedPoint operator*(FixedPoint other) const;
    FixedPoint operator/(FixedPoint other) const;

    bool operator==(FixedPoint other) const;
    bool operator!=(FixedPoint other) const;
    bool operator<(FixedPoint other) const;
    bool operator>(FixedPoint other) const;
    bool operator<=(FixedPoint other) const;
    bool operator>=(FixedPoint other) const;

private:
    std::int32_t raw_;
};

using Fixed16 = FixedPoint;

} // namespace cnn

#endif
