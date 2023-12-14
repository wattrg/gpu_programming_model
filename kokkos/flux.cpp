#include <Kokkos_Core.hpp>

int N = 500000;
const double R = 287.0;
const int n_repeats = 1000;
double Cv = 5.0 / 2.0 * R;
double Cp = 7.0 / 2.0 * R;
double gam_ = Cp / Cv;

typedef Kokkos::View<double*> Array;

struct FlowStates {
  FlowStates() : 
    p(Array("p", N)), 
    T(Array("T", N)), 
    rho(Array("rho", N)), 
    vx(Array("vx", N)), 
    vy(Array("vy", N)), 
    vz(Array("vz", N)) {}

  Array p;
  Array T;
  Array rho;
  Array vx;
  Array vy;
  Array vz;
};

struct Conserved {
  Conserved () :
    mass(Array("mass", N)),
    px(Array("px", N)),
    py(Array("py", N)),
    pz(Array("pz", N)),
    e(Array("e", N)) {}
  Array mass;
  Array px;
  Array py;
  Array pz;
  Array e;
};

void ausmdv(const FlowStates &left, const FlowStates &right, Conserved &flux) {
  double R_ = R;
  double Cv_ = Cv;
  double gamma_ = gam_;
  Kokkos::parallel_for("ausmdv", N, KOKKOS_LAMBDA(const int i) {
      double rL = left.rho(i);
      double pL = left.p(i);
      double pLrL = pL / rL;
      double uL = left.vx(i);
      double vL = left.vy(i);
      double wL = left.vz(i);
      double eL = Cv_ * left.T(i); //gm.internal_energy(left.gas, i);
      double aL = Kokkos::sqrt(gamma_ * R_ * left.T(i)); // gm.speed_of_sound(left.gas, i);
      double keL = 0.5 * (uL * uL + vL * vL + wL * wL);
      double HL = eL + pLrL + keL;


      double rR = right.rho(i);
      double pR = right.p(i);
      double pRrR = pR / rR;
      double uR = right.vx(i);
      double vR = right.vy(i);
      double wR = right.vz(i);
      double eR = Cv_ * right.T(i); //gm.internal_energy(right.gas, i);
      double aR = Kokkos::sqrt(gamma_ * R_ * right.T(i)); // gm.speed_of_sound(right.gas, i);
      double keR = 0.5 * (uR * uR + vR * vR + wR * wR);
      double HR = eR + pRrR + keR;

      // This is the main part of the flux calculator.
      // Weighting parameters (eqn 32) for velocity splitting.
      double alphaL = 2.0 * pLrL / (pLrL + pRrR);
      double alphaR = 2.0 * pRrR / (pLrL + pRrR);

      // Common sound speed (eqn 33) and Mach doubles.
      double am = Kokkos::fmax(aL, aR);
      double ML = uL / am;
      double MR = uR / am;

      // Left state:
      // pressure splitting (eqn 34)
      // and velocity splitting (eqn 30)
      double pLplus, uLplus;
      double duL = 0.5 * (uL + Kokkos::fabs(uL));
      if (Kokkos::fabs(ML) <= 1.0) {
        pLplus = pL * (ML + 1.0) * (ML + 1.0) * (2.0 - ML) * 0.25;
        uLplus = alphaL * ((uL + am) * (uL + am) / (4.0 * am) - duL) + duL;
      } else {
        pLplus = pL * duL / uL;
        uLplus = duL;
      }

      // Right state:
      // pressure splitting (eqn 34)
      // and velocity splitting (eqn 31)
      double pRminus, uRminus;
      double duR = 0.5 * (uR - Kokkos::fabs(uR));
      if (Kokkos::fabs(MR) <= 1.0) {
        pRminus = pR * (MR - 1.0) * (MR - 1.0) * (2.0 + MR) * 0.25;
        uRminus = alphaR * (-(uR - am) * (uR - am) / (4.0 * am) - duR) + duR;
      } else {
        pRminus = pR * duR / uR;
        uRminus = duR;
      }

      // Mass Flux (eqn 29)
      // The mass flux is relative to the moving interface.
      double ru_half = uLplus * rL + uRminus * rR;

      // Pressure flux (eqn 34)
      double p_half = pLplus + pRminus;

      // Momentum flux: normal direction
      // Compute blending parameter s (eqn 37),
      // the momentum flux for AUSMV (eqn 21) and AUSMD (eqn 21)
      // and blend (eqn 36).
      double dp = pL - pR;
      const double K_SWITCH = 10.0;
      dp = K_SWITCH * Kokkos::fabs(dp) / Kokkos::fmin(pL, pR);
      double s = 0.5 * Kokkos::fmin(1.0, dp);
      double ru2_AUSMV = uLplus * rL * uL + uRminus * rR * uR;
      double ru2_AUSMD = 
       0.5 * (ru_half * (uL + uR) - Kokkos::fabs(ru_half) * (uR - uL));
      double ru2_half = (0.5 + s) * ru2_AUSMV + (0.5 - s) * ru2_AUSMD;

      // Assemble components of the flux vector.
      flux.mass(i) = ru_half;
      if (ru_half >= 0.0) {
        // Wind is blowing from the left.
        flux.px(i) = (ru2_half + p_half);
        flux.py(i) = (ru_half * vL);
        flux.pz(i) = (ru_half * wL);
        flux.e(i) = ru_half * HL;
      } else {
        // Wind is blowing from the right.
        flux.px(i) = (ru2_half + p_half);
        flux.py(i) = (ru_half * vR);
        flux.pz(i) = (ru_half * wR);
        flux.e(i) = ru_half * HR;
      }
  }); 
}

int main() {
  double time;
  Kokkos::initialize();
  {
    FlowStates left {};
    FlowStates right {};
    Conserved flux {};

    Array::HostMirror hpl = Kokkos::create_mirror(left.p);
    Array::HostMirror hTl = Kokkos::create_mirror(left.T);
    Array::HostMirror hrl = Kokkos::create_mirror(left.rho);
    Array::HostMirror hvxl = Kokkos::create_mirror(left.vx);
    Array::HostMirror hvyl = Kokkos::create_mirror(left.vy);
    Array::HostMirror hvzl = Kokkos::create_mirror(left.vz);

    Array::HostMirror hpr = Kokkos::create_mirror(right.p);
    Array::HostMirror hTr = Kokkos::create_mirror(right.T);
    Array::HostMirror hrr = Kokkos::create_mirror(right.rho);
    Array::HostMirror hvxr = Kokkos::create_mirror(right.vx);
    Array::HostMirror hvyr = Kokkos::create_mirror(right.vy);
    Array::HostMirror hvzr = Kokkos::create_mirror(right.vz);
    for (int i = 0; i < N; ++i) {

      hpl(i) = 101325.0;
      hTl(i) = 300.0;
      hrl(i) = hpl(i) / ( R * hTl(i));
      hvxl(i) = 4000 / N * i - 2000;
      hvyl(i) = 4000 / N * i - 2000;
      hvzl(i) = 4000 / N * i - 2000;

      hpr(i) = 101325.0;
      hTr(i) = 300.0;
      hrr(i) = hpr(i) / ( R * hTr(i));
      hvxr(i) = 4000 / N * i - 2000;
      hvyr(i) = 4000 / N * i - 2000;
      hvzr(i) = 4000 / N * i - 2000;
    }

    Kokkos::deep_copy(left.p, hpl);
    Kokkos::deep_copy(left.T, hTl);
    Kokkos::deep_copy(left.rho, hrl);
    Kokkos::deep_copy(left.vx, hvxl);
    Kokkos::deep_copy(left.vy, hvyl);
    Kokkos::deep_copy(left.vz, hvzl);

    Kokkos::deep_copy(right.p, hpr);
    Kokkos::deep_copy(right.T, hTr);
    Kokkos::deep_copy(right.rho, hrr);
    Kokkos::deep_copy(right.vx, hvxr);
    Kokkos::deep_copy(right.vy, hvyr);
    Kokkos::deep_copy(right.vz, hvzr);

    Kokkos::Timer timer;

    for (int repeat = 0; repeat < n_repeats; repeat++){
      ausmdv(left, right, flux);
    }

    Kokkos::fence();

    time = timer.seconds();

    // printf("%f\n", left.p(0));
  }
  Kokkos::finalize();
  printf("time = %g ms\n", time * 1e3 / n_repeats);
}
