
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace unit {
  template <bool p, typename T>
  struct enable_if {
     using type = T;   
  };
  template <typename T>
  struct enable_if<false, T> {

  };
 
  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  class Quantity;

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  std::ostream& operator<<(std::ostream&, const Quantity<R, m, kg, s, K, A, mol, cd>&);

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  class Quantity {
  public:
    Quantity() :value{0.0} {} 
    Quantity(R value) :value{value} {}
    friend std::ostream& operator<< <> (std::ostream& os, const Quantity& quantity);
    R value;
  };
  
  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  std::ostream& operator<<(std::ostream& os, const Quantity<R, m, kg, s, K, A, mol, cd>& quantity)
  {
    os << quantity.value;
    return os;
  }

  /* Overloading + for Quantity addition with itself and R */
  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator+(Quantity<R, m, kg, s, K, A, mol, cd> op1,
	    Quantity<R, m, kg, s, K, A, mol, cd> op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1.value + op2.value};
  }


  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator+(Quantity<R, m, kg, s, K, A, mol, cd> op1,
	    R op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1.value + op2};
  }

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator+(R op1,
	    Quantity<R, m, kg, s, K, A, mol, cd> op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op2.value + op1};
  }

  /* Overloading - for Quantity subtraction with itself and R */
  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator-(Quantity<R, m, kg, s, K, A, mol, cd> op1,
	    Quantity<R, m, kg, s, K, A, mol, cd> op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1.value - op2.value};
  }

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator-(Quantity<R, m, kg, s, K, A, mol, cd> op1,
	    R op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1.value - op2};
  }

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator-(R op1,
	    Quantity<R, m, kg, s, K, A, mol, cd> op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1 - op2.value};
  }

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator-(Quantity<R, m, kg, s, K, A, mol, cd> op)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{-op.value};
  }
  
  /* Overloading * for Quantity subtraction with itself and R */
  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator*(Quantity<R, m, kg, s, K, A, mol, cd> op1, R op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1.value * op2};
  }

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator*(R op1, Quantity<R, m, kg, s, K, A, mol, cd> op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1 * op2.value};
  }

  template < typename R,
	     int m1, int kg1, int s1, int K1, int A1, int mol1, int cd1,
	     int m2, int kg2, int s2, int K2, int A2, int mol2, int cd2 >
  typename enable_if<!(m1 == -m2 &&
		       kg1 == -kg2 &&
		       s1 == -s2 &&
		       K1 == -K2 &&
		       A1 == -A2 &&
		       mol1 == -mol2 &&
		       cd1 == -cd2),
		     Quantity<R, m1 + m2, kg1 + kg2, s1 + s2, K1 + K2, A1 + A2, mol1 + mol2, cd1 + cd2>>::type
  operator*(Quantity<R, m1, kg1, s1, K1, A1, mol1, cd1> q1, Quantity<R, m2, kg2, s2, K2, A2, mol2, cd2> q2)
  {
    return Quantity<R, m1 + m2, kg1 + kg2, s1 + s2, K1 + K2, A1 + A2, mol1 + mol2, cd1 + cd2>{q1.value * q2.value};
  }

  
  /* Overloading / for Quantity subtraction with itself and R */
  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, m, kg, s, K, A, mol, cd>
  operator/(Quantity<R, m, kg, s, K, A, mol, cd> op1,
	    R op2)
  {
    return Quantity<R, m, kg, s, K, A, mol, cd>{op1.value / op2};
  }

  template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  Quantity<R, -m, -kg, -s, -K, -A, -mol, -cd>
  operator/(R op1,
	    Quantity<R, m, kg, s, K, A, mol, cd> op2)
  {
    return Quantity<R, -m, -kg, -s, -K, -A, -mol, -cd>{op1 / op2.value};
  }

  template < typename R,
	     int m1, int kg1, int s1, int K1, int A1, int mol1, int cd1,
	     int m2, int kg2, int s2, int K2, int A2, int mol2, int cd2 >
  typename enable_if<!(m1 == m2 &&
		       kg1 == kg2 &&
		       s1 == s2 &&
		       K1 == K2 &&
		       mol1 == mol2 &&
		       cd1 == cd2), 
		     Quantity<R, m1 - m2, kg1 - kg2, s1 - s2, K1 - K2, A1 - A2, mol1 - mol2, cd1 - cd2>>::type
  operator/(Quantity<R, m1, kg1, s1, K1, A1, mol1, cd1> q1, Quantity<R, m2, kg2, s2, K2, A2, mol2, cd2> q2)
  {
    return Quantity<R, m1 - m2, kg1 - kg2, s1 - s2, K1 - K2, A1 - A2, mol1 - mol2, cd1 - cd2>{q1.value / q2.value};
  } 
  
  /* type transformations */
  template <typename T, typename R>
  struct QuotientUnit {
    T op1;
    R op2;
    using type = decltype(op1 / op2);
  };

  template <typename T, typename R>
  struct ProductUnit {
    T op1;
    R op2;
    using type = decltype(op1 * op2);
  };

  template <typename T>
  struct Inverse {
    T op;
    using type = decltype(1.0 / op);
  };

  using dimensionless = Quantity<double, 0, 0, 0, 0, 0, 0, 0>;
  using Volt    = Quantity<double, 2, 1, -3, 0, -1, 0, 0>;
  using Newton  = Quantity<double, 1, 1, -2, 0, 0, 0, 0>;
  using meter   = Quantity<double, 1, 0, 0, 0, 0, 0, 0>;
  using Joule   = ProductUnit<Newton, meter>::type;
  using Ampere  = Quantity<double, 0, 0, 0, 0, 1, 0, 0>;
  using second  = Quantity<double, 0, 0, 1, 0, 0, 0, 0>;
  using Farad   = Quantity<double, -2, -1, 4, 0, 2, 0, 0>;
  using Siemens = Quantity<double, -2, -1, 3, 0, 2, 0, 0>;
  using Volt_per_second = QuotientUnit<Volt, second>::type;
  // using Hertz = Quantity<double, 0, 0, -1, 0, 0, 0, 0>;
  using Hertz = Inverse<second>::type;

  /* allowing for type coercions */
  template <typename T, typename E>
  constexpr T CoerceTo(E expr);


  template <typename T, typename R, int m, int kg, int s, int K, int A, int mol, int cd>
  constexpr T CoerceTo(Quantity<R, m, kg, s, K, A, mol, cd> expr)
  {
    return T{expr.value};
  }
  
  template <typename T>
  constexpr T CoerceTo(double value)
  {
    return T{value};
  }

  template <typename P, int e>
  class CreatePowerFunction {
  private:
    P function;
  public:
    template <typename R, int m, int kg, int s, int K, int A, int mol, int cd>
    Quantity<R, m * e, kg * e, s * e, K * e, A * e, mol * e, cd * e>
    operator() (Quantity<R, m, kg, s, K, A, mol, cd> base)
    {
      return function(base.value, e);
    }
  };

  enum class Arity {
    One,
    Two,
    Three,
    Four
  };

  template <typename M, Arity arity>
  class CreateNumericFunction {
    /* meaningless without arity */
  };

  template <typename M>
  class CreateNumericFunction <M, Arity::One> {
  private:
    M function;
  public:
    dimensionless operator() (dimensionless arg)
    {
      return dimensionless{function(arg.value)};
    }
  };

  template <typename M>
  class CreateNumericFunction <M, Arity::Two> {
  private:
    M function;
  public:
    dimensionless operator() (dimensionless arg, dimensionless arg2)
    {
      return dimensionless{function(arg.value, arg2.value)};
    }
  };

  template <typename M>
  class CreateNumericFunction <M, Arity::Three> {
  private:
    M function;
  public:
    dimensionless operator() (dimensionless arg, dimensionless arg2, dimensionless arg3)
    {
      return dimensionless{function(arg.value, arg2.value, arg3.value)};
    }
  };

  template <typename M>
  class CreateNumericFunction <M, Arity::Four> {
  private:
    M function;
  public:
    dimensionless operator() (dimensionless arg, dimensionless arg2, dimensionless arg3, dimensionless arg4)
    {
      return dimensionless{function(arg.value, arg2.value, arg3.value, arg4.value)};
    }
  };
}

namespace unit_aux {
  /* note: allow user to create custom power functors using inheritance */
  class Power {
  public:
    double operator()(double base, int exponent)
    {
      return pow(base, static_cast<double>(exponent));
    }
  };

  class Exp {
  public:
    double operator()(double exponent)
    {
      return exp(exponent);
    }
  };

  unit::CreatePowerFunction<Power, 3> pow3;
  unit::CreatePowerFunction<Power, 4> pow4;
  unit::CreateNumericFunction<Exp, unit::Arity::One> exp;
}

using namespace unit;
using namespace unit_aux;

using real = double;

Ampere const I = 10.0;
Farad const C = 1.0;
Siemens const g_Na = 120.0;
Siemens const g_K = 36.0;
Siemens const g_L = 0.3;
Volt const E_Na = 120.0;
Volt const E_K = -12.0;
Volt const E_L = 10.6;

class Neuron {
public:
  enum class DerivativeName {
    dV_wrt_dt = 0,
    dn_wrt_dt,
    dm_wrt_dt,
    dh_wrt_dt
  };
  Neuron(Volt V, dimensionless n, dimensionless m, dimensionless h);
  Volt_per_second dV_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h);
  Hertz dn_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h);
  Hertz dm_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h);
  Hertz dh_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h);
  void update(second time_step);
  Volt V;
  dimensionless n, m, h;
};

Neuron::Neuron(Volt V, dimensionless n, dimensionless m, dimensionless h)
  :V{V}, n{n}, m{m}, h{h}
{
}

Volt_per_second
Neuron::dV_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h)
{ 
  return (I - g_K*unit_aux::pow4(n)*(V - E_K) - g_Na*unit_aux::pow3(m)*h*(V - E_Na) - g_L*(V - E_L)) / C; 
}


Hertz alpha_n(Volt V)
{
  return CoerceTo<Hertz>(0.01 * (10.0 - V)/(unit_aux::exp(CoerceTo<dimensionless>((10.0 - V)/10.0)) - 1.0));
}

Hertz alpha_m(Volt V)
{
  return CoerceTo<Hertz>(0.1 * (25.0 - V)/(unit_aux::exp(CoerceTo<dimensionless>((25.0 - V)/10.0)) - 1.0));
}

Hertz alpha_h(Volt V)
{
  return CoerceTo<Hertz>(0.07 * unit_aux::exp(CoerceTo<dimensionless>(-V/20.0)));
}

Hertz beta_n(Volt V)
{
  return CoerceTo<Hertz>(0.125 * unit_aux::exp(CoerceTo<dimensionless>(-V/80.0)));
}

Hertz beta_m(Volt V)
{
  return CoerceTo<Hertz>(4.0 * unit_aux::exp(CoerceTo<dimensionless>(-V/18.0)));
}

Hertz beta_h(Volt V)
{
  return CoerceTo<Hertz>(1.0 / (unit_aux::exp(CoerceTo<dimensionless>((30.0 - V)/10.0)) + 1.0));
}

Hertz Neuron::dn_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h)
{
  return alpha_n(V) * (1.0 - n) - beta_n(V)*n;
}

Hertz Neuron::dm_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h)
{
  return alpha_m(V) * (1.0 - m) - beta_m(V)*m;
}

Hertz Neuron::dh_wrt_dt(Volt V, dimensionless n, dimensionless m, dimensionless h)
{
  return alpha_h(V) * (1.0 - h) - beta_h(V)*h;
}

void Neuron::update(second time_step)
{
  struct intermediate {
    Volt V;
    dimensionless n, m, h;
  };

  intermediate k1, k2, k3, k4, t1, t2, t3;

  k1.V = time_step * dV_wrt_dt(V, n, m, h); 
  k1.n = time_step * dn_wrt_dt(V, n, m, h);
  k1.m = time_step * dm_wrt_dt(V, n, m, h);
  k1.h = time_step * dh_wrt_dt(V, n, m, h);

  t1.V = V + k1.V/2.0;
  t1.n = n + k1.n/2.0;
  t1.m = m + k1.m/2.0;
  t1.h = h + k1.h/2.0;

  k2.V = time_step * dV_wrt_dt(t1.V, t1.n, t1.m, t1.h);
  k2.n = time_step * dn_wrt_dt(t1.V, t1.n, t1.m, t1.h);
  k2.m = time_step * dm_wrt_dt(t1.V, t1.n, t1.m, t1.h);
  k2.h = time_step * dh_wrt_dt(t1.V, t1.n, t1.m, t1.h);

  t2.V = V + k2.V/2.0;
  t2.n = n + k2.n/2.0;
  t2.m = m + k2.m/2.0;
  t2.h = h + k2.h/2.0;
  
  k3.V = time_step * dV_wrt_dt(t2.V, t2.n, t2.m, t2.h);
  k3.n = time_step * dn_wrt_dt(t2.V, t2.n, t2.m, t2.h);
  k3.m = time_step * dm_wrt_dt(t2.V, t2.n, t2.m, t2.h);
  k3.h = time_step * dh_wrt_dt(t2.V, t2.n, t2.m, t2.h);

  t3.V = V + k3.V;
  t3.n = n + k3.n;
  t3.m = m + k3.m;
  t3.h = h + k3.h;
	
  k4.V = time_step * dV_wrt_dt(t3.V, t3.n, t3.m, t3.h);
  k4.n = time_step * dn_wrt_dt(t3.V, t3.n, t3.m, t3.h);
  k4.m = time_step * dm_wrt_dt(t3.V, t3.n, t3.m, t3.h);
  k4.h = time_step * dh_wrt_dt(t3.V, t3.n, t3.m, t3.h);

  V = V + k1.V/6.0 + k2.V/3.0 + k3.V/3.0 + k4.V/6.0;
  n = n + k1.n/6.0 + k2.n/3.0 + k3.n/3.0 + k4.n/6.0;
  m = m + k1.m/6.0 + k2.m/3.0 + k3.m/3.0 + k4.m/6.0;
  h = h + k1.h/6.0 + k2.h/3.0 + k3.h/3.0 + k4.h/6.0;
}

int main()
{ 
  const int number_of_points = 100000;
  std::ofstream output("hh-extended2.dat");
  output << std::setiosflags(std::ios::showpoint | std::ios::uppercase);

  second time_step = 0.01;
  Neuron first_neuron{Volt{65.0}, dimensionless{0.0}, dimensionless{0.0}, dimensionless{0.0}};
  
  output << std::setw(15) << std::setprecision(8) << 0.0 << '\t' << first_neuron.V << std::endl;
  for (int i = 0; i < number_of_points; i++) {
    first_neuron.update(time_step);
    output << std::setw(15) << std::setprecision(8) << (i + 1.0) * time_step << '\t' << first_neuron.V << std::endl;
  }

  return 0;
}
