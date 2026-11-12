#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* Profiling results (as of July 2022) indicate that about 75% of total
   execution time is spent in the easy_dgtsv function and the rest in the
   _implicit_time function.  Profiling performed with the "yep" python module. */

/* We want to avoid depending on lapack so this is an implementation of the
   Thomas algorithm, which performs the same function as dgtsv in lapack.  It
   modifies the coefficients to garbage values, but the return value is saved in
   b. */

void easy_dgtsv(int n, double *dl, double *d, double *du, double *b) {
  if (n==1) {
    b[0] /= d[0];
    return;
  }
  double tmp;
  int i;
  for (i=1; i<n; i++) {
    tmp = dl[i-1]/d[i-1];
    d[i] -= tmp*du[i-1];
    b[i] -= tmp*b[i-1];
  }
  b[n-1] = b[n-1]/d[n-1];
  for (i=n-2; i>=0; i--) {
    b[i] = (b[i] - du[i]*b[i+1])/d[i];
  }
}


double* _analytic_ddm_linbound(double a1, double b1, double a2, double b2, unsigned int nsteps, double tstep);
int _implicit_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr, double* drift, double* noise, double *bound, double *ic, int Xsteps, double dt, double dx, unsigned int drift_mode, unsigned int noise_mode, unsigned int bound_mode, double mass_tolerance);
int _cn_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr, double* drift, double* noise, double *bound, double *ic, int Xsteps, double dt, double dx, unsigned int drift_mode, unsigned int noise_mode, unsigned int bound_mode, double mass_tolerance);

static PyObject* analytic_ddm_linbound(PyObject* self, PyObject* args) {
  double a1, b1, a2, b2, tstep;
  int nsteps;
  if (!PyArg_ParseTuple(args, "ddddid", &a1, &b1, &a2, &b2, &nsteps, &tstep))
    return NULL;
  double *res = _analytic_ddm_linbound(a1, b1, a2, b2, nsteps, tstep);
  npy_intp dims[1] = { nsteps };
  PyObject *retarray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, res);
  PyArray_ENABLEFLAGS((PyArrayObject*)retarray, NPY_ARRAY_OWNDATA);
  return retarray;
}


static PyObject* implicit_time(PyObject* self, PyObject* args) {
  double dt, dx, T_dur;
  double mass_tolerance = 1e-14;
  double *drift, *noise, *bound, *ic;
  PyArrayObject *_drift, *_noise, *_bound, *_ic;
  PyObject *__drift, *__noise, *__bound, *__ic;
  int nsteps, len_x0;
  int drifttype, noisetype, boundtype;
  if (!PyArg_ParseTuple(args, "OiOiOiOdddi|d", &__drift, &drifttype, &__noise, &noisetype, &__bound, &boundtype, &__ic, &T_dur, &dt, &dx, &nsteps, &mass_tolerance))
    return NULL;

  /* TODO here is the memory leak: if you don't call the FROMANY function, there is no leak */
  _drift = (PyArrayObject*)PyArray_FROMANY(__drift, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
  _noise = (PyArrayObject*)PyArray_FROMANY(__noise, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
  _bound = (PyArrayObject*)PyArray_FROMANY(__bound, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
  _ic = (PyArrayObject*)PyArray_FROMANY(__ic, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

  len_x0 = PyArray_SIZE(_ic);
  if (!_drift || !_noise || !_bound || !_ic)
    return NULL;
  drift = (double*)PyArray_DATA(_drift);
  noise = (double*)PyArray_DATA(_noise);
  bound = (double*)PyArray_DATA(_bound);
  ic = (double*)PyArray_DATA(_ic);


  double *pdfchoice1 = (double*)malloc(nsteps*sizeof(double));
  double *pdfchoice2 = (double*)malloc(nsteps*sizeof(double));
  double *pdfcurr = (double*)malloc(len_x0*sizeof(double));
  _implicit_time(nsteps, pdfchoice1, pdfchoice2, pdfcurr, drift, noise, bound, ic, len_x0, dt, dx, drifttype, noisetype, boundtype, mass_tolerance);
  npy_intp dims[1] = { nsteps };
  npy_intp dimscurr[1] = { len_x0 };
  PyObject *choice1array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, pdfchoice1);
  PyObject *choice2array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, pdfchoice2);
  PyObject *currarray = PyArray_SimpleNewFromData(1, dimscurr, NPY_DOUBLE, pdfcurr);
  /* Need ENABLEFLAGS here, not UpdateFlags, because UpdateFlags checks to make
     sure you aren't using flags it doesn't like and silently drops those flags.
     (OWNDATA is one such flag.)  ENABLEFLAGS doesn't perform this validation. */
  PyArray_ENABLEFLAGS((PyArrayObject*)choice1array, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*)choice2array, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*)currarray, NPY_ARRAY_OWNDATA);

  /* Get rid of the reference to the PyArray arguments, without this there is a memory leak */
  Py_DECREF(_drift);
  Py_DECREF(_noise);
  Py_DECREF(_bound);
  Py_DECREF(_ic);

  /* Use NNN instead of OOO because OOO increments the reference count. */
  PyObject *ret = Py_BuildValue("(NNN)", choice1array, choice2array, currarray);
  return ret;
}


static PyObject* cn_time(PyObject* self, PyObject* args) {
  double dt, dx, T_dur;
  double mass_tolerance = 1e-14;
  double *drift, *noise, *bound, *ic;
  PyArrayObject *_drift, *_noise, *_bound, *_ic;
  PyObject *__drift, *__noise, *__bound, *__ic;
  int nsteps, len_x0;
  int drifttype, noisetype, boundtype;
  if (!PyArg_ParseTuple(args, "OiOiOiOdddi|d", &__drift, &drifttype, &__noise, &noisetype, &__bound, &boundtype, &__ic, &T_dur, &dt, &dx, &nsteps, &mass_tolerance))
    return NULL;

  _drift = (PyArrayObject*)PyArray_FROMANY(__drift, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
  _noise = (PyArrayObject*)PyArray_FROMANY(__noise, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
  _bound = (PyArrayObject*)PyArray_FROMANY(__bound, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
  _ic = (PyArrayObject*)PyArray_FROMANY(__ic, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);

  len_x0 = PyArray_SIZE(_ic);
  if (!_drift || !_noise || !_bound || !_ic)
    return NULL;
  drift = (double*)PyArray_DATA(_drift);
  noise = (double*)PyArray_DATA(_noise);
  bound = (double*)PyArray_DATA(_bound);
  ic = (double*)PyArray_DATA(_ic);

  double *pdfchoice1 = (double*)malloc(nsteps*sizeof(double));
  double *pdfchoice2 = (double*)malloc(nsteps*sizeof(double));
  double *pdfcurr = (double*)malloc(len_x0*sizeof(double));
  _cn_time(nsteps, pdfchoice1, pdfchoice2, pdfcurr, drift, noise, bound, ic, len_x0, dt, dx, drifttype, noisetype, boundtype, mass_tolerance);
  npy_intp dims[1] = { nsteps };
  npy_intp dimscurr[1] = { len_x0 };
  PyObject *choice1array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, pdfchoice1);
  PyObject *choice2array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, pdfchoice2);
  PyObject *currarray = PyArray_SimpleNewFromData(1, dimscurr, NPY_DOUBLE, pdfcurr);
  PyArray_ENABLEFLAGS((PyArrayObject*)choice1array, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*)choice2array, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject*)currarray, NPY_ARRAY_OWNDATA);

  Py_DECREF(_drift);
  Py_DECREF(_noise);
  Py_DECREF(_bound);
  Py_DECREF(_ic);

  PyObject *ret = Py_BuildValue("(NNN)", choice1array, choice2array, currarray);
  return ret;
}


/*  define functions in module */
static PyMethodDef DDMMethods[] =
{
     {"analytic_ddm_linbound", analytic_ddm_linbound, METH_VARARGS, "DDM with linear bound"},
     {"implicit_time", implicit_time, METH_VARARGS, "DDM with implicit method"},
     {"cn_time", cn_time, METH_VARARGS, "DDM with Crank-Nicolson method"},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module =
{
    PyModuleDef_HEAD_INIT,
    "csolve", "Some documentation",
    -1,
    DDMMethods
};

PyMODINIT_FUNC
PyInit_csolve(void)
{
    import_array();
    PyObject *mod = PyModule_Create(&module);
    PyModule_AddIntConstant(mod, "CONSTANT_TX", 0);
    PyModule_AddIntConstant(mod, "CHANGING_T", 1);
    PyModule_AddIntConstant(mod, "CHANGING_X", 2);
    PyModule_AddIntConstant(mod, "CHANGING_TX", 3);
    return mod;
}


/* Methods */

double* _analytic_ddm_linbound(double a1, double b1, double a2, double b2, unsigned int nsteps, double tstep) {
  const int nMax = 100; /* Maximum numbe of loops */
  const double errbnd = 1e-10; /* Error bound for looping */
  double* suminc;
  double* tmp;
  int checkerr = 0;
  int i,n;
  const double pi = 3.14159265358979323846;
  const double width = a1-a2;
  int image_steps = nsteps;
  double eigen_coefficient[8], eigen_rate = 0;
  /* Parallel bounds admit a rapidly converging sine expansion at long
     times.  Keep the image series at short times, where it is faster and
     avoids cancellation.  At t/width^2 >= .2 the omitted ninth sine term
     is below 2e-34 before the common density prefactor. */
  if (b1 == b2 && a1 > 0 && a2 < 0) {
    double transition = .2*width*width/tstep;
    if (transition < nsteps) {
      image_steps = (int)ceil(transition);
      if (image_steps < 1) image_steps = 1;
      eigen_rate = pi*pi/(2*width*width);
      for (n=1; n<=8; n++)
        eigen_coefficient[n-1] = pi/(width*width)*n*sin(pi*n*a1/width);
    }
  }

  /* Prepare suminc */
  suminc = (double*)malloc(sizeof(double)*nsteps);
  memset(suminc, 0, nsteps*sizeof(double));

  /* The two image exponents are quadratic in n. Their successive ratios
     therefore have a constant ratio themselves. Preserve the same sum and
     stopping test, but evaluate exp only during this initialization. */
  tmp = (double*)malloc(5*(size_t)image_steps*sizeof(double));
  double *first = tmp+image_steps, *second = tmp+2*image_steps;
  double *ratio_first = tmp+3*image_steps, *ratio_second = tmp+4*image_steps;
  for (i=1; i<image_steps; i++) {
    double exponent = -2.*(width/(i*tstep)+b1-b2);
    tmp[i] = exp(2*width*exponent);
    first[i] = 1;
    second[i] = exp(-a2*exponent);
    ratio_first[i] = exp((a1+width)*exponent);
    ratio_second[i] = exp((2*width-a2)*exponent);
  }
  double maxval;
  double toadd;
  double na1a2;
  double coef2, coef4;
  checkerr = 0;
  for (n=0; n<nMax; n++) {
    maxval = 0;
    na1a2 = n*width;
    coef2 = a1+2*na1a2;
    coef4 = a1-2*a2+2*na1a2;
    for (i=1; i<image_steps; i++) {
      toadd = first[i]*coef2-second[i]*coef4;
      suminc[i] += toadd;
      first[i] *= ratio_first[i];
      second[i] *= ratio_second[i];
      ratio_first[i] *= tmp[i];
      ratio_second[i] *= tmp[i];
      double relative = fabs(toadd / (suminc[i] == 0 ? 1e-30 : suminc[i]));
      if (relative > maxval)
        maxval = relative;
      }
    if (maxval < errbnd) {
      if (++checkerr >= 3)
        break;
    } else {
      checkerr = 0;
    }
  }
  /*const double sqrtpi = sqrt(2*M_PI); */
  const double oneoversqrtpi = 1/sqrt(2*pi);
  /* float itstep; */
  for (i=1; i<image_steps; i++) {
    /* suminc[i] *= exp(-pow((a1+b1*i*tstep),2)/(i*tstep)/2)/sqrtpi/pow(i*tstep, 1.5); */
    /* itstep = i*tstep; */
    suminc[i] *= oneoversqrtpi*exp(-.5*(a1+b1*i*tstep)*(a1+b1*i*tstep)/(i*tstep))*pow(i*tstep, -1.5);
    if (suminc[i] < 0)
      suminc[i] = 0;
  }
  for (i=image_steps; i<nsteps; i++) {
    double t = i*tstep;
    double drift_exponent = -b1*a1-.5*b1*b1*t;
    /* exp(-a*n^2) has successive ratios exp(-a*(2*n+1)).  Generate
       those ratios by multiplication rather than eight calls to exp. */
    double base = exp(-eigen_rate*t);
    double ratio_step = base*base;
    double ratio = base*ratio_step;
    double term = exp(drift_exponent-eigen_rate*t);
    for (n=1; n<=8; n++) {
      suminc[i] += eigen_coefficient[n-1]*term;
      term *= ratio;
      ratio *= ratio_step;
    }
    if (suminc[i] < 0) suminc[i] = 0;
  }
  suminc[0] = 0;
  free(tmp);
  return suminc;
}



/* All time integrators use the same conservative spatial operator.  Its
   diagonal is -L, its off-diagonals are negative transition rates, and
   lower/upper are the rates leaving the two adjacent interior points. */
typedef struct {
  double *diag, *up, *down;
  double lower, upper;
  int first, last;
} FPMatrix;

static FPMatrix fp_alloc(int n) {
  FPMatrix m;
  m.diag = (double*)calloc(n, sizeof(double));
  m.up = (double*)calloc(n, sizeof(double));
  m.down = (double*)calloc(n, sizeof(double));
  m.lower = m.upper = 0;
  m.first = m.last = 0;
  return m;
}

static void fp_free(FPMatrix m) {
  free(m.diag);
  free(m.up);
  free(m.down);
}

static int fp_radius(double bound, double dx, int centre) {
  int radius = (int)ceil(bound/dx - 1e-10)-1;
  if (radius < 0) radius = 0;
  if (radius > centre) radius = centre;
  return radius;
}

static void fp_absorb(double *pdf, int n, double bound, double dx,
                      double *lower, double *upper) {
  int centre = (n-1)/2;
  int radius = fp_radius(bound, dx, centre);
  int first = centre-radius, last = centre+radius;
  *lower = *upper = 0;
  if (bound <= 0) {
    first = last = centre;
    *lower = *upper = .5*pdf[centre];
    pdf[centre] = 0;
  }
  for (int j=0; j<first; j++) {
    *lower += pdf[j];
    pdf[j] = 0;
  }
  for (int j=last+1; j<n; j++) {
    *upper += pdf[j];
    pdf[j] = 0;
  }
}

static void fp_matrix(FPMatrix *m, int n, double dx, double bound,
                      double *drift, double *noise, int time,
                      unsigned int drift_mode, unsigned int noise_mode) {
  int centre = (n-1)/2;
  int radius = fp_radius(bound, dx, centre);
  int dmultx = (drift_mode & 2) ? 1 : 0;
  int nmultx = (noise_mode & 2) ? 1 : 0;
  int dmultt = (drift_mode & 1) ? (dmultx ? n : 1) : 0;
  int nmultt = (noise_mode & 1) ? (nmultx ? n : 1) : 0;
  double dxinv = 1/dx;
  memset(m->diag, 0, n*sizeof(double));
  memset(m->up, 0, n*sizeof(double));
  memset(m->down, 0, n*sizeof(double));
  m->first = centre-radius;
  m->last = centre+radius;
  m->lower = m->upper = 0;
  if (bound <= 0) return;
  for (int j=m->first; j<=m->last; j++) {
    double mu = drift[time*dmultt+j*dmultx];
    double sigma = noise[time*nmultt+j*nmultx];
    double diffusion = .5*sigma*sigma*dxinv*dxinv;
    double advection = .5*mu*dxinv;
    double rate_left = diffusion-advection, rate_right = diffusion+advection;
    if (j == m->first || j == m->last) {
      double x = (j-centre)*dx;
      double left = j == m->first ? bound+x : dx;
      double right = j == m->last ? bound-x : dx;
      rate_left = (sigma*sigma-mu*right)/(left*(left+right));
      rate_right = (sigma*sigma+mu*left)/(right*(left+right));
    }
    m->diag[j] = rate_left+rate_right;
    if (j == m->first) m->lower = rate_left;
    else m->up[j-1] = -rate_left;
    if (j == m->last) m->upper = rate_right;
    else m->down[j] = -rate_right;
  }
}

static void fp_step(double *pdf, int factor_cache, FPMatrix old, FPMatrix next,
                    double step, double theta, double *diag, double *up,
                    double *down, double *rhs, double *lower, double *upper) {
  *lower = step*(1-theta)*old.lower*pdf[old.first];
  *upper = step*(1-theta)*old.upper*pdf[old.last];
  int first = next.first, last = next.last;
  if (theta == 1) {
    rhs = pdf;  /* BE needs no old-state contribution after this point. */
  } else {
    for (int j=first; j<=last; j++) {
      double applied = old.diag[j]*pdf[j];
      if (j > first) applied += old.down[j-1]*pdf[j-1];
      if (j < last) applied += old.up[j]*pdf[j+1];
      rhs[j] = pdf[j] - step*(1-theta)*applied;
    }
  }
  /* Cache a constant tridiagonal factorization only within this solve:
     0 = changing matrix, 1 = initialize factors, 2 = reuse factors.
     CN and its BE startup half steps have identical left-hand matrices. */
  if (factor_cache != 2) {
    for (int j=first; j<=last; j++) {
      diag[j] = 1 + step*theta*next.diag[j];
      up[j] = step*theta*next.up[j];
      down[j] = step*theta*next.down[j];
    }
  }
  if (factor_cache == 0) {
    easy_dgtsv(last-first+1, down+first, diag+first, up+first, rhs+first);
  } else {
    if (factor_cache == 1) {
      diag[first] = 1/diag[first];
      for (int j=first+1; j<=last; j++) {
        down[j-1] *= diag[j-1];
        diag[j] = 1/(diag[j]-down[j-1]*up[j-1]);
      }
    }
    for (int j=first+1; j<=last; j++) rhs[j] -= down[j-1]*rhs[j-1];
    rhs[last] *= diag[last];
    for (int j=last-1; j>=first; j--) rhs[j] = (rhs[j]-up[j]*rhs[j+1])*diag[j];
  }
  *lower += step*theta*next.lower*rhs[next.first];
  *upper += step*theta*next.upper*rhs[next.last];
  if (rhs != pdf)
    memcpy(pdf+first, rhs+first, (last-first+1)*sizeof(double));
}

static int fp_time(int Tsteps, double *pdfchoice1, double *pdfchoice2,
                   double *pdfcurr, double *drift, double *noise, double *bound,
                   double *ic, int Xsteps, double dt, double dx,
                   unsigned int drift_mode, unsigned int noise_mode,
                   unsigned int bound_mode, int cn, double mass_tolerance) {
  FPMatrix old = fp_alloc(Xsteps), next = fp_alloc(Xsteps), middle = fp_alloc(Xsteps);
  double *diag = (double*)malloc(Xsteps*sizeof(double));
  double *up = (double*)malloc(Xsteps*sizeof(double));
  double *down = (double*)malloc(Xsteps*sizeof(double));
  double *rhs = (double*)malloc(Xsteps*sizeof(double));
  double out_lower, out_upper, swept_lower, swept_upper;
  int last_cn_step = 0;
  int varying = (drift_mode & 1) || (noise_mode & 1) || bound_mode;
  memset(pdfchoice1, 0, Tsteps*sizeof(double));
  memset(pdfchoice2, 0, Tsteps*sizeof(double));
  memcpy(pdfcurr, ic, Xsteps*sizeof(double));
  fp_absorb(pdfcurr, Xsteps, bound[0], dx, &pdfchoice2[0], &pdfchoice1[0]);
  fp_matrix(&old, Xsteps, dx, bound[0], drift, noise, 0, drift_mode, noise_mode);
  if (!varying)
    fp_matrix(&next, Xsteps, dx, bound[0], drift, noise, 0, drift_mode, noise_mode);
  for (int i=1; i<Tsteps; i++) {
    if ((i-1) % 10 == 0) {
      double remaining = 0;
      for (int j=old.first; j<=old.last; j++) remaining += fabs(pdfcurr[j]);
      if (remaining == 0 || remaining < mass_tolerance) break;
    }
    double b = bound[bound_mode ? i : 0];
    int factor_cache = varying ? 0 : (i == 1 ? 1 : 2);
    if (varying)
      fp_matrix(&next, Xsteps, dx, b, drift, noise, i, drift_mode, noise_mode);
    if (cn) {
      if (i <= 2) {
        /* Four backward Euler half steps damp the initial point source.
           Interpolate the operator, not sigma, at each first half step. */
        for (int j=0; j<Xsteps; j++) {
          middle.diag[j] = .5*(old.diag[j]+next.diag[j]);
          middle.up[j] = .5*(old.up[j]+next.up[j]);
          middle.down[j] = .5*(old.down[j]+next.down[j]);
        }
        middle.first = old.first;
        middle.last = old.last;
        middle.lower = .5*(old.lower+next.lower);
        middle.upper = .5*(old.upper+next.upper);
        fp_step(pdfcurr, factor_cache, old, middle, .5*dt, 1,
                diag, up, down, rhs, &out_lower, &out_upper);
        double half_lower = out_lower, half_upper = out_upper;
        fp_step(pdfcurr, varying ? 0 : 2, middle, next, .5*dt, 1,
                diag, up, down, rhs, &out_lower, &out_upper);
        /* Centre this interval's absorbed mass on the existing time grid. */
        pdfchoice1[i] = .5*(half_upper+out_upper);
        pdfchoice2[i] = .5*(half_lower+out_lower);
        pdfchoice1[i-1] += pdfchoice1[i];
        pdfchoice2[i-1] += pdfchoice2[i];
      } else {
        if (i == 3) {
          /* Complete the last startup node with the old-state contribution
             from the first trapezoidal interval. */
          pdfchoice1[i-1] += .5*dt*old.upper*pdfcurr[old.last];
          pdfchoice2[i-1] += .5*dt*old.lower*pdfcurr[old.first];
        }
        fp_step(pdfcurr, factor_cache, old, next, dt, .5,
                diag, up, down, rhs, &out_lower, &out_upper);
        /* Interior nodes receive two equal trapezoidal half weights. */
        pdfchoice1[i] = dt*next.upper*pdfcurr[next.last];
        pdfchoice2[i] = dt*next.lower*pdfcurr[next.first];
      }
      last_cn_step = i;
    } else {
      swept_lower = swept_upper = 0;
      if (bound_mode)
        fp_absorb(pdfcurr, Xsteps, b, dx, &swept_lower, &swept_upper);
      if (b <= 0) {
        pdfchoice1[i] = swept_upper;
        pdfchoice2[i] = swept_lower;
        break;
      }
      fp_step(pdfcurr, factor_cache, next, next, dt, 1,
              diag, up, down, rhs, &out_lower, &out_upper);
      pdfchoice1[i] = swept_upper+out_upper;
      pdfchoice2[i] = swept_lower+out_lower;
    }
    FPMatrix swap = old;
    old = next;
    next = swap;
  }
  if (last_cn_step > 2) {
    /* No absorption after the last integrated time, including early exits. */
    pdfchoice1[last_cn_step] *= .5;
    pdfchoice2[last_cn_step] *= .5;
  }
  /* Preserve the extension API: implicit_time returns densities, cn_time
     returns probability masses on the time grid. */
  if (!cn) {
    for (int i=0; i<Tsteps; i++) {
      pdfchoice1[i] /= dt;
      pdfchoice2[i] /= dt;
    }
  }
  fp_free(old);
  fp_free(next);
  fp_free(middle);
  free(diag);
  free(up);
  free(down);
  free(rhs);
  return 0;
}

int _implicit_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr,
                   double *drift, double *noise, double *bound, double *ic, int Xsteps,
                   double dt, double dx, unsigned int drift_mode, unsigned int noise_mode,
                   unsigned int bound_mode, double mass_tolerance) {
  return fp_time(Tsteps, pdfchoice1, pdfchoice2, pdfcurr, drift, noise, bound, ic,
                 Xsteps, dt, dx, drift_mode, noise_mode, bound_mode, 0, mass_tolerance);
}

int _cn_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr,
             double *drift, double *noise, double *bound, double *ic, int Xsteps,
             double dt, double dx, unsigned int drift_mode, unsigned int noise_mode,
             unsigned int bound_mode, double mass_tolerance) {
  return fp_time(Tsteps, pdfchoice1, pdfchoice2, pdfcurr, drift, noise, bound, ic,
                 Xsteps, dt, dx, drift_mode, noise_mode, bound_mode, 1, mass_tolerance);
}
