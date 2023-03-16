# fast-learn-sagemath

## Sage Quick Reference

### Notebook
```
Evaluate cell: Shift-Enter
Evaluate cell creating new cell: Alt-Enter
Split cell: Control-;
Insert cell: mouse move to  the line between cells, then click +code or +text
```

### Command line
```
com<tab> complete command
*bar*? list command names containing “bar”
command?<tab> shows documentation
command??<tab> shows source code
a.<tab> shows methods for object a (more: dir(a))
a._<tab> shows hidden methods for object a
search_doc("string or regexp") fulltext search of docs
search_src("string or regexp") search source code
_ is previous output
```

### Numbers
```
ZZ QQ RR CC 
Double precision:RDF CDF
CC(1,1)
CDF(2.1,3)
Mod n: Z/nZ = Zmod e.g. Mod(2,3) Zmod(3)(2)
Finite fields: Fq = GF e.g. GF(3)(2) GF(9,"a").0
Polynomials: R[x, y] e.g. S.<x,y>=QQ[] x+2*y^3
Series: R[[t]] e.g. S.<t>=QQ[[]] 1/2+2*t+O(t^2)
p-adic numbers: Zp ≈Zp, Qp ≈Qp e.g. 2+3*5+O(5^2)
Algebraic closure: Q = QQbar e.g. QQbar(2^(1/5))
Interval arithmetic: RIF e.g. sage: RIF((1,1.00001))
Number field: R.<x>=QQ[];K.<a>=NumberField(x^3+x+1)
```

###  Arithmetic Constants functions
```
a*b a/b a^b sqrt(x) x^(1/n) abs(x) log(x,b)
sum(f(i) for i in (k..n))
prod(f(i) for i in (k..n))
Constants: π = pi e = e i = i ∞ = oo
φ = golden_ratio γ = euler_gamma
Approximate: pi.n(digits=18) = 3.14159265358979324
Functions: sin cos tan sec csc cot sinh cosh tanh
sech csch coth log ln exp ...
Python function: 
def f(x): 
  return x^2
```

### Interactive functions
```
# Put @interact before function (vars determine controls)
@interact
def f(n=[0..4], s=(1..5), c=Color("red")):
  var("x");
  show(plot(sin(n+x^s),-pi,pi,color=c))
```
### Symbolic expressions
```
Define new symbolic variables: var("t u v y z")
Symbolic function: e.g. f(x)=x^2
Relations: f==g f<=g f>=g f<g f>g
Solve f = g: solve(f(x)==g(x), x)
solve([f(x,y)==0, g(x,y)==0], x,y)
factor(...) expand(...) (...).simplify_...
find_root(f(x), a, b) find x ∈ [a, b] s.t. f(x) ≈ 0
```
### Calculus
```
limit(f(x), x=a)
diff(f(x),x)
diff(f(x,y),x)
diff = differentiate = derivative
integral(f(x),x)
integral(f(x),x,a,b)
numerical_integral(f(x),a,b)
Taylor polynomial, deg n about a: taylor(f(x),x,a,n)
var('x,k,n');
taylor (sqrt (1 - k^2*sin(x)^2), x, 0, 6)
```
### 2D graphics
```
line([(x1,y1),. . .,(xn,yn)],options)
polygon([(x1,y1),. . .,(xn,yn)],options)
circle((x,y),r,options)
text("txt",(x,y),options)
options as in plot.options, e.g. thickness=pixel,
rgbcolor=(r,g,b), hue=h where 0 ≤ r, b, g, h ≤ 1
show(graphic, options)
use figsize=[w,h] to adjust size
use aspect_ratio=number to adjust aspect ratio
plot(f(x),(x, xmin, xmax),options)
parametric_plot((f(t),g(t)),(t, tmin, tmax),options)
polar_plot(f(t),(t, tmin, tmax),options)
combine: circle((1,1),1)+line([(0,0),(2,2)])
animate(list of graphics, options).show(delay=20)
```

```
g=line([(0,1),(1,2),(3,3)],color=Color("green"),axes=False)+polygon([(-1,1),(0,2),(3,3)],color=Color("lightblue"),axes=False)+circle((0,1.25),0.17,color=Color("orange"),axes=False)
show(g)
```
### 3D graphics
```
line3d([(x1,y1,z1),. . .,(xn,yn,zn)],options)
sphere((x,y,z),r,options)
text3d("txt", (x,y,z), options)
tetrahedron((x,y,z),size,options)
cube((x,y,z),size,options)
octahedron((x,y,z),size,options)
dodecahedron((x,y,z),size,options)
icosahedron((x,y,z),size,options)
plot3d(f(x, y),(x, xb, xe), (y, yb, ye),options)
parametric_plot3d((f,g,h),(t, tb, te),options)
parametric_plot3d((f(u, v),g(u, v),h(u, v)),
(u, ub, ue),(v, vb, ve),options)
options: aspect_ratio=[1, 1, 1], color="red"
opacity=0.5, figsize=6, viewer="tachyon"
```

### Discrete math
```
floor(x) ceil(x) n%k k|n iff n%k==0
n! = factorial(n) binomial(x,m) φ(n)=euler_phi(n)
Strings: e.g. s = "Hello" = "He"+’llo’
s[0]="H" s[-1]="o" s[1:3]="el" s[3:]="lo"
Lists: e.g. [1,"Hello",x] = []+[1,"Hello"]+[x]
Tuples: e.g. (1,"Hello",x) (immutable)
Sets: e.g. {1, 2, 1, a} = Set([1,2,1,"a"]) (= {1, 2, a})
List comprehension ≈ set builder notation, e.g.
{f(x) : x ∈ X, x > 0} = Set([f(x) for x in X if x>0])
```

### Graph theory
```
Graph: G = Graph({0:[1,2,3], 2:[4]})
Directed Graph: DiGraph(dictionary)
Graph families: graphs.<tab>
Invariants: G.chromatic_polynomial(), G.is_planar()
Paths: G.shortest_path()
Visualize: G.plot(), G.plot3d()
Automorphisms: G.automorphism_group(),
G1.is_isomorphic(G2), G1.is_subgraph(G2)
```

### Combinatorics
```
Integer sequences: sloane_find(list), sloane.<tab>
Partitions: P=Partitions(n) P.count()
Combinations: C=Combinations(list) C.list()
Cartesian product: CartesianProduct(P,C)
Tableau: Tableau([[1,2,3],[4,5]])
Words: W=Words("abc"); W("aabca")
Posets: Poset([[1,2],[4],[3],[4],[]])
Root systems: RootSystem(["A",3])
Crystals: CrystalOfTableaux(["A",3], shape=[3,2])
Lattice Polytopes: A=random_matrix(ZZ,3,6,x=7)
L=LatticePolytope(A) L.npoints() L.plot3d()
```
### Matrix algebra & Linear algebra
```
vector([1,2])
matrix(QQ,[[1,2],[3,4]], sparse=False)
matrix(QQ,2,3,[1,2,3, 4,5,6])
det(matrix(QQ,[[1,2],[3,4]]))
A*v  A^-1 A.transpose()
Solve Ax = v: A\v or A.solve_right(v)
Solve xA = v: A.solve_left(v)
Reduced row echelon form: A.echelon_form()
Rank and nullity: A.rank() A.nullity()
Hessenberg form: A.hessenberg_form()
Characteristic polynomial: A.charpoly()
Eigenvalues: A.eigenvalues()
Eigenvectors: A.eigenvectors_right() (also left)
Gram-Schmidt: A.gram_schmidt()
Visualize: A.plot()
LLL reduction: matrix(ZZ,...).LLL()
Hermite form: matrix(ZZ,...).hermite_form()

Vector space Kn = K^n e.g. QQ^3 RR^2 CC^4
Subspace: span(vectors, field )
E.g., span([[1,2,3], [2,3,5]], QQ)
Kernel: A.right_kernel() (also left)
Sum and intersection: V + W and V.intersection(W)
Basis: V.basis()
Basis matrix: V.basis_matrix()
Restrict matrix to subspace: A.restrict(V)
Vector in terms of basis: V.coordinates(vector)
```
### Numerical mathematics
```
Packages: import numpy, scipy, cvxopt
Minimization: var("x y z")
minimize(x^2+x*y^3+(1-z)^2-1, [1,1,1])
```

### Number theory
```
gcd(n,m), gcd(list)
extended gcd g = sa + tb = gcd(a, b): g,s,t=xgcd(a,b)
lcm(n,m), lcm(list)
digits in a given base: n.digits(base)
number of digits: n.ndigits(base)
divides n | m: n.divides(m) if nk = m some k
factorial – n! = n.factorial()
n.divisors()
set of prime numbers: Primes()
{p : m ≤ p < n and p prime} =prime_range(n,m)
prime powers: prime_powers(m,n)
first n primes: primes_first_n(n)
is_prime, is_pseudoprime is_prime_power next_prime prime_pi
previous_prime(n), next_probable_prime(n) next_prime_power(n), pevious_prime_power(n)
Factor: factor(n), qsieve(n), ecm.factor(n)
binomial(m,n) kronecker(-1,5) kronecker_symbol(a,b) continued_fraction(x) bernoulli(n), Bernoulli_mod_p(p)
EllipticCurve([a1, a2, a3, a4, a6]) DirichletGroup(N) ModularForms(level, weight) ModularSymbols(level, weight, sign)
BrandtModule(level, weight)
Modular abelian varieties: J0(N), J1(N)
vector(range(1,5)) # (1,2,3,4)
vector(1..3) # (1,2,3)

euler_phi(n) kronecker_symbol(a,b) quadratic_residues(n)

ring Z/nZ = Zmod(n) = IntegerModRing(n)
a modulo n as element of Z/nZ: Mod(a, n)
primitive_root(n)
inverse of n (mod m): n.inverse_mod(m)
power a^n (mod m): power_mod(a, n, m)
Chinese remainder theorem: x = crt([a,b],[m,n])
finds x with x ≡ a (mod m) and x ≡ b (mod n)
discrete log: log(Mod(6,7), Mod(3,7))
order of a (mod n) : Mod(a,n).multiplicative_order()
square root of a (mod n) : Mod(a,n).sqrt()
```
```
Lucas-Lehmer test for primality of 2p − 1
def is_prime_lucas_lehmer(p):
s = Mod(4, 2^p - 1)
for i in range(3, p+1): s = s^2 - 2
return s == 0
```

```
k=12; m = matrix(ZZ, k, [(i*j)%k for i in [0..k-1] for j in [0..k-1]]); m.plot(cmap='gray')
```

#### Special Functions
```
zeta Li gamma beta(1/2,1/2).n()
complex_plot(zeta, (-30,5), (-8,8))
```

#### Continued Fractions
```
continued fraction: c=continued_fraction(x, bits)
convergents: c.convergents()
convergent numerator pn = c.pn(n)
convergent denominator qn = c.qn(n)
value: c.value()
pi==[3;7,15,1,292,...]
```

### Group theory
```
G = PermutationGroup([[(1,2,3),(4,5)],[(3,4)]])
SymmetricGroup(n), AlternatingGroup(n)
Abelian groups: AbelianGroup([3,15])
Matrix groups: GL, SL, Sp, SU, GU, SO, GO
Functions: G.sylow_subgroup(p), G.character_table(),
G.normal_subgroups(), G.cayley_graph()
```
### Noncommutative rings
```
Quaternions: Q.<i,j,k> = QuaternionAlgebra(a,b)
Free algebra: R.<a,b,c> = FreeAlgebra(QQ, 3)
```

### Python modules
```
import module_name
module_name.<tab> and help(module_name)
```
### Profiling and debugging
``` 
time command: show timing information
timeit("command"): accurately time command
t = cputime(); cputime(t): elapsed CPU time
t = walltime(); walltime(t): elapsed wall time
%pdb: turn on interactive debugger (command line only)
%prun command: profile command (command line only)
```

### Elliptic Curves
...
