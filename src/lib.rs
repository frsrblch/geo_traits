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

pub trait ConstGradeProduct<Rhs, const N: u32> {
    type Output;
    fn product(self, rhs: Rhs) -> Self::Output;
}

pub fn product<Lhs, Rhs, Output, const N: u32>(lhs: Lhs, rhs: Rhs) -> Output
where
    Lhs: ConstGradeProduct<Rhs, N, Output = Output>,
{
    <Lhs as ConstGradeProduct<Rhs, N>>::product(lhs, rhs)
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

/// The sandwich product: ABA<sup>-1</sup>
pub trait Sandwich<Rhs> {
    type Output;
    fn sandwich(self, rhs: Rhs) -> Self::Output;
}

/// The sandwich product: A⟇B⟇A<sup>-1</sup>
pub trait Antisandwich<Rhs> {
    type Output;
    fn antisandwich(self, rhs: Rhs) -> Self::Output;
}

/// For implementing generic traits and functions over types with an inner type parameter
pub trait FloatType {
    type Float;
}

pub trait Map<U>: FloatType {
    type Output: FloatType<Float = U>;
    fn map<F: Fn(Self::Float) -> U>(self, f: F) -> Self::Output;
}

pub trait SubsetOf<T> {
    fn from_superset(value: T) -> Self;
}
