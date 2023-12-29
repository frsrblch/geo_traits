//! Traits for geometric algebra

pub use num_traits::{one, zero, FloatConst, Inv, One, Zero};

/// A minimal set of traits that serve as a lightweight alternative to `num_traits::Float`.
///
/// `num_traits::Float` is a one-for-one stand-in for `f32` and `f64`,
/// and precludes other types that may not implement their full interface.
///
/// By contrast, `Number` descibes a the functionality required for a numeric type to be used
/// as an approximation of the real numbers in geometric algebra.
///
/// A practical example is the use of dual numbers to perform automatic differentiation and differentiable programming.
pub trait Number:
    Sized
    + Copy
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Sqrt<Output = Self>
    + Ln<Output = Self>
    + Exp<Output = Self>
    + Trig<Output = Self>
    + InvTrig<Output = Self>
    + Inv<Output = Self>
    + Zero
    + One
    + FromF64
{
}

impl<T> Number for T where
    T: Sized
        + Copy
        + PartialEq
        + PartialOrd
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Div<Output = Self>
        + std::ops::Neg<Output = Self>
        + Sqrt<Output = Self>
        + Ln<Output = Self>
        + Exp<Output = Self>
        + Trig<Output = Self>
        + InvTrig<Output = Self>
        + Inv<Output = Self>
        + Zero
        + One
        + FromF64
{
}

/// Similar to `Number`, but allows binary operations between heterogeneous types.
///
/// A practical example is the addition of floats and dual numbers. The float could be converted to a dual number
/// before the operation, but the compiler applying the correct optimizations is not guaranteed.
pub trait Numbers<U: Number>:
    Number
    + std::ops::Add<U, Output = <Self as Numbers<U>>::Output>
    + std::ops::Sub<U, Output = <Self as Numbers<U>>::Output>
    + std::ops::Mul<U, Output = <Self as Numbers<U>>::Output>
    + std::ops::Div<U, Output = <Self as Numbers<U>>::Output>
{
    type Output: Number;
}

impl<T, U, V> Numbers<U> for T
where
    T: Number
        + std::ops::Add<U, Output = V>
        + std::ops::Sub<U, Output = V>
        + std::ops::Mul<U, Output = V>
        + std::ops::Div<U, Output = V>,
    U: Number,
    V: Number,
{
    type Output = V;
}

pub trait Pow<F = Self> {
    type Output;
    fn pow(self, power: F) -> Self::Output;
}

impl Pow for f32 {
    type Output = f32;
    #[inline]
    fn pow(self, power: f32) -> Self::Output {
        f32::powf(self, power)
    }
}

impl Pow for f64 {
    type Output = f64;

    fn pow(self, power: f64) -> Self::Output {
        f64::powf(self, power)
    }
}

/// Convertion from `f64` for generic code that requires hard-coded constants
pub trait FromF64 {
    fn from_f64(value: f64) -> Self;
}

impl FromF64 for f32 {
    #[inline]
    fn from_f64(value: f64) -> f32 {
        value as f32
    }
}

impl FromF64 for f64 {
    #[inline]
    fn from_f64(value: f64) -> f64 {
        value
    }
}

/// Trigonometric and hyperbolic trigonometric functions
pub trait Trig {
    type Output;
    const TAU: Self;
    fn sin(self) -> Self::Output;
    fn cos(self) -> Self::Output;
    fn sin_cos(self) -> (Self::Output, Self::Output);
    fn tan(self) -> Self::Output;
    fn sinh(self) -> Self::Output;
    fn cosh(self) -> Self::Output;
    fn tanh(self) -> Self::Output;
}

/// Inverse trigonometric and hyperbolic trigonometric functions
pub trait InvTrig {
    type Output;
    fn asin(self) -> Self::Output;
    fn acos(self) -> Self::Output;
    fn atan(self) -> Self::Output;
    fn atan2(self, x: Self) -> Self::Output;
    fn asinh(self) -> Self::Output;
    fn acosh(self) -> Self::Output;
    fn atanh(self) -> Self::Output;
}

macro_rules! impl_fn {
    ($fn_:ident) => {
        #[inline]
        fn $fn_(self) -> Self::Output {
            self.$fn_()
        }
    };
    ($($fn_:ident),* $(,)?) => {
        $(
            impl_fn!($fn_);
        )*
    };
}

impl Trig for f32 {
    const TAU: Self = std::f32::consts::TAU;

    type Output = f32;

    impl_fn!(sin, cos, tan, sinh, cosh, tanh);

    #[inline]
    fn sin_cos(self) -> (Self::Output, Self::Output) {
        self.sin_cos()
    }
}

impl InvTrig for f32 {
    type Output = f32;

    impl_fn!(asin, acos, atan, asinh, acosh, atanh);

    #[inline]
    fn atan2(self, x: Self) -> Self::Output {
        self.atan2(x)
    }
}

impl Trig for f64 {
    const TAU: Self = std::f64::consts::TAU;

    type Output = f64;

    impl_fn!(sin, cos, tan, sinh, cosh, tanh);

    #[inline]
    fn sin_cos(self) -> (Self::Output, Self::Output) {
        self.sin_cos()
    }
}

impl InvTrig for f64 {
    type Output = f64;

    impl_fn!(asin, acos, atan, asinh, acosh, atanh);

    #[inline]
    fn atan2(self, x: Self) -> Self::Output {
        self.atan2(x)
    }
}

/// The geometric product: A⟑B
pub trait Geo<Rhs> {
    type Output;
    fn geo(self, rhs: Rhs) -> Self::Output;
}

/// The exterior product: A∧B
pub trait Wedge<Rhs> {
    type Output;
    fn wedge(self, rhs: Rhs) -> Self::Output;
}

/// The inner product: A⋅B = Σ〈〈A〉<sub>r</sub>〈B〉<sub>s</sub>〉<sub>|r-s|</sub>
pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

/// The antigeometric product: A⟇B
pub trait Antigeo<Rhs> {
    type Output;
    fn antigeo(self, rhs: Rhs) -> Self::Output;
}

/// The antidot product: A○B
pub trait Antidot<Rhs> {
    type Output;
    fn antidot(self, rhs: Rhs) -> Self::Output;
}

/// The regressive product: A∨B
pub trait Antiwedge<Rhs> {
    type Output;
    fn antiwedge(self, rhs: Rhs) -> Self::Output;
}

/// The commutator product: A×B = ½(AB - BA)
pub trait Commutator<Rhs> {
    type Output;
    fn com(self, rhs: Rhs) -> Self::Output;
}

/// The left contraction: A ⌋ B = Σ〈〈A〉<sub>r</sub>〈B〉<sub>s</sub>〉<sub>r-s</sub>
pub trait LeftContraction<Rhs> {
    type Output;
    fn left_con(self, rhs: Rhs) -> Self::Output;
}

/// The right contraction: A ⌊ B = Σ〈〈A〉<sub>r</sub>〈B〉<sub>s</sub>〉<sub>s-r</sub>
pub trait RightContraction<Rhs> {
    type Output;
    fn right_con(self, rhs: Rhs) -> Self::Output;
}

/// The geometric product of grade n: 〈AB〉<sub>n</sub>
pub trait GradeProduct<Lhs, Rhs> {
    type Output;
    fn product(lhs: Lhs, rhs: Rhs) -> Self::Output;
}

/// The geometric antiproduct of grade n: 〈A⟇B〉<sub>n</sub>
pub trait GradeAntiproduct<Lhs, Rhs> {
    type Output;
    fn antiproduct(lhs: Lhs, rhs: Rhs) -> Self;
}

/// The dual function: A(A*) = I
pub trait Dual {
    type Output;
    fn dual(self) -> Self::Output;
}

/// The left complement: A̲ such that A̲A = ||A||<sup>2</sup> I
pub trait LeftComplement {
    type Output;
    fn left_comp(self) -> Self::Output;
}

/// The right complement: Ā such that AĀ = ||A||<sup>2</sup> I
pub trait RightComplement {
    type Output;
    fn right_comp(self) -> Self::Output;
}

/// The reverse function: Ã
pub trait Reverse {
    type Output;
    fn rev(self) -> Self::Output;
}

/// The antireverse function
pub trait Antireverse {
    type Output;
    fn antirev(self) -> Self;
}

/// The grade involution function
pub trait GradeInvolution {
    type Output;
    fn grade_involution(self) -> Self;
}

/// The clifford conjugate, the combination of reversion and grade involution
pub trait CliffordConjugate {
    type Output;
    fn conjugate(self) -> Self::Output;
}

/// The norm squared function: ||A||<sup>2</sup> = AÃ
pub trait Norm2 {
    type Output;
    fn norm2(self) -> Self::Output;
}

/// The norm function: ||A|| = √(AÃ)
pub trait Norm {
    type Output;
    fn norm(self) -> Self::Output;
}

/// The unitize function: Â = A / ||A||
pub trait Unitize {
    type Output;
    fn unit(self) -> Self::Output;
}

/// The sandwich product: ABÃ
pub trait Sandwich<Rhs> {
    type Output;
    fn sandwich(self, rhs: Rhs) -> Self::Output;
}

/// The sandwich product: A⟇B⟇Ã
pub trait Antisandwich<Rhs> {
    type Output;
    fn antisandwich(self, rhs: Rhs) -> Self::Output;
}

/// For implementing traits and functions over types with a generic inner type parameter
pub trait FloatType {
    type Float;
}

/// For implementing traits and functions over types with a generic magnitude
pub trait MagnitudeType {
    type Mag;
}

/// For implementing traits and functions over types with a generic magnitude
pub trait MagnitudeAssert<M> {
    type Output;
    fn assert(self) -> Self::Output;
}

/// Map the values of a multivector directly
pub trait Map<U = <Self as FloatType>::Float>: FloatType {
    type Output;
    fn map<F: Fn(Self::Float) -> U>(self, f: F) -> Self::Output;
}

macro_rules! impl_float_fn_trait {
    ($trait_:ident :: $fn_:ident $(, $doc:literal)?) => {
        $(#[doc = $doc])?
        pub trait $trait_ {
            type Output;
            fn $fn_(self) -> Self::Output;
        }

        impl $trait_ for f32 {
            type Output = f32;
            #[inline]
            fn $fn_(self) -> Self::Output {
                f32::$fn_(self)
            }
        }

        impl $trait_ for f64 {
            type Output = f64;
            #[inline]
            fn $fn_(self) -> Self::Output {
                f64::$fn_(self)
            }
        }
    };
}

impl_float_fn_trait!(Sqrt::sqrt, "The square root function");
impl_float_fn_trait!(Cbrt::cbrt, "The cubic root function");
impl_float_fn_trait!(Ln::ln, "The natural logarithm: ln(R)");
impl_float_fn_trait!(Exp::exp, "The exponential function: eᴮ");

/// The quartic root function. Implemented as a double square root.
pub trait Qrt {
    type Output;
    fn qrt(self) -> Self::Output;
}

impl Qrt for f32 {
    type Output = f32;
    #[inline]
    fn qrt(self) -> Self::Output {
        self.sqrt().sqrt()
    }
}

impl Qrt for f64 {
    type Output = f64;
    #[inline]
    fn qrt(self) -> Self::Output {
        self.sqrt().sqrt()
    }
}

/// A const one value
pub trait OneConst {
    const ONE: Self;
    #[inline]
    fn is_one(&self) -> bool
    where
        Self: Sized + PartialEq,
    {
        Self::ONE.eq(self)
    }
}

macro_rules! one_const {
    ($ty:ty: $one:expr) => {
        impl OneConst for $ty {
            const ONE: Self = $one;
        }
    };
    ($($ty:ty: $one:expr,)*) => {
        $(
            one_const!($ty: $one);
        )*
    };
}

one_const! {
    f32: 1.0,
    f64: 1.0,
    u8: 1,
    u16: 1,
    u32: 1,
    u64: 1,
    u128: 1,
    i8: 1,
    i16: 1,
    i32: 1,
    i64: 1,
    i128: 1,
}

/// A const zero value
pub trait ZeroConst {
    const ZERO: Self;
    #[inline]
    fn is_one(&self) -> bool
    where
        Self: Sized + PartialEq,
    {
        Self::ZERO.eq(self)
    }
}

macro_rules! zero_const {
    ($ty:ty: $zero:expr) => {
        impl ZeroConst for $ty {
            const ZERO: Self = $zero;
        }
    };
    ($($ty:ty: $zero:expr,)*) => {
        $(
            zero_const!($ty: $zero);
        )*
    };
}

zero_const! {
    f32: 0.0,
    f64: 0.0,
    u8: 0,
    u16: 0,
    u32: 0,
    u64: 0,
    u128: 0,
    i8: 0,
    i16: 0,
    i32: 0,
    i64: 0,
    i128: 0,
}

/// Base 10 logarithm
pub trait Log10 {
    type Output;
    fn log10(self) -> Self::Output;
}

impl Log10 for f32 {
    type Output = f32;
    #[inline]
    fn log10(self) -> Self::Output {
        self.log10()
    }
}

impl Log10 for f64 {
    type Output = f64;
    #[inline]
    fn log10(self) -> Self::Output {
        self.log10()
    }
}

/// Base 10 exponentiation
pub trait Exp10 {
    type Output;
    fn exp10(self) -> Self::Output;
}

impl Exp10 for f32 {
    type Output = f32;
    #[inline]
    fn exp10(self) -> Self::Output {
        10f32.powf(self)
    }
}

impl Exp10 for f64 {
    type Output = f64;
    #[inline]
    fn exp10(self) -> Self::Output {
        10f64.powf(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_impl_number() {
        fn impls_number<T: Number>(_: &T) {}

        impls_number(&0f32);
        impls_number(&0f64);
    }

    #[test]
    fn floats_impl_numbers() {
        fn impls_numbers<T: Numbers<T, Output = T>>(_: &T) {}

        impls_numbers(&0f32);
        impls_numbers(&0f64);
    }

    #[test]
    fn log_mass_test() {}
}
