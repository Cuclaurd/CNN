#include "../include/fixpoint.h"

#include <limits>

namespace cnn {

FixedPoint::FixedPoint()
    : raw_(0) {}

FixedPoint::FixedPoint(int value)
    : raw_(static_cast<std::int32_t>(value * scale)) {}

FixedPoint::FixedPoint(float value)
    : raw_(static_cast<std::int32_t>(value * scale)) {}

FixedPoint::FixedPoint(double value)
    : raw_(static_cast<std::int32_t>(value * scale)) {}

FixedPoint FixedPoint::from_raw(std::int32_t raw) {
    FixedPoint result;
    result.raw_ = raw;
    return result;
}

std::int32_t FixedPoint::raw() const {
    return raw_;
}

float FixedPoint::to_float() const {
    return static_cast<float>(raw_) / scale;
}

double FixedPoint::to_double() const {
    return static_cast<double>(raw_) / scale;
}

FixedPoint FixedPoint::operator-() const {
    if (raw_ == std::numeric_limits<std::int32_t>::min()) {
        return from_raw(std::numeric_limits<std::int32_t>::max());
    }
    return from_raw(static_cast<std::int32_t>(-raw_));
}

FixedPoint FixedPoint::operator+(FixedPoint other) const {
    return from_raw(static_cast<std::int32_t>(raw_ + other.raw_));
}

FixedPoint FixedPoint::operator-(FixedPoint other) const {
    return from_raw(static_cast<std::int32_t>(raw_ - other.raw_));
}

FixedPoint FixedPoint::operator*(FixedPoint other) const {
    std::int64_t result =
        static_cast<std::int64_t>(raw_) *
        static_cast<std::int64_t>(other.raw_);
    result /= scale;
    return from_raw(static_cast<std::int32_t>(result));
}

FixedPoint FixedPoint::operator/(FixedPoint other) const {
    if (other.raw_ == 0) {
        return from_raw(0);
    }

    std::int64_t result =
        static_cast<std::int64_t>(raw_) *
        static_cast<std::int64_t>(scale);
    result /= other.raw_;
    return from_raw(static_cast<std::int32_t>(result));
}

FixedPoint& FixedPoint::operator+=(FixedPoint other) {
    raw_ = static_cast<std::int32_t>(raw_ + other.raw_);
    return *this;
}

FixedPoint& FixedPoint::operator-=(FixedPoint other) {
    raw_ = static_cast<std::int32_t>(raw_ - other.raw_);
    return *this;
}

bool FixedPoint::operator==(FixedPoint other) const {
    return raw_ == other.raw_;
}

bool FixedPoint::operator!=(FixedPoint other) const {
    return raw_ != other.raw_;
}

bool FixedPoint::operator<(FixedPoint other) const {
    return raw_ < other.raw_;
}

bool FixedPoint::operator>(FixedPoint other) const {
    return raw_ > other.raw_;
}

bool FixedPoint::operator<=(FixedPoint other) const {
    return raw_ <= other.raw_;
}

bool FixedPoint::operator>=(FixedPoint other) const {
    return raw_ >= other.raw_;
}

} // namespace cnn
