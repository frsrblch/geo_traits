//! Traits for geometric algebra

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
pub trait Map<U>: FloatType {
    type Output: FloatType<Float = U>;
    fn map<F: Fn(Self::Float) -> U>(self, f: F) -> Self::Output;
}

/// The square root function
pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}

impl Sqrt for f32 {
    type Output = f32;
    #[inline]
    fn sqrt(self) -> Self::Output {
        f32::sqrt(self)
    }
}

impl Sqrt for f64 {
    type Output = f64;
    #[inline]
    fn sqrt(self) -> Self::Output {
        f64::sqrt(self)
    }
}

/// The natural logarithm: ln(R)
pub trait Ln {
    type Output;
    fn ln(self) -> Self::Output;
}

impl Ln for f32 {
    type Output = f32;
    #[inline]
    fn ln(self) -> Self::Output {
        f32::ln(self)
    }
}

impl Ln for f64 {
    type Output = f64;
    #[inline]
    fn ln(self) -> Self::Output {
        f64::ln(self)
    }
}

/// The exponential function: eᴮ
pub trait Exp {
    type Output;
    fn exp(self) -> Self::Output;
}

impl Exp for f32 {
    type Output = f32;

    fn exp(self) -> Self::Output {
        f32::exp(self)
    }
}

impl Exp for f64 {
    type Output = f64;

    fn exp(self) -> Self::Output {
        f64::exp(self)
    }
}

/// A const one value
pub trait OneConst {
    const ONE: Self;
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
