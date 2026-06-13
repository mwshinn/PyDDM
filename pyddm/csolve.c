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
    d[0] = b[0]/d[0];
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
int _implicit_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr, double* drift, double* noise, double *bound, double *ic, int Xsteps, double dt, double dx, unsigned int drift_mode, unsigned int noise_mode, unsigned int bound_mode);
int _cn_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr, double* drift, double* noise, double *bound, double *ic, int Xsteps, double dt, double dx, unsigned int drift_mode, unsigned int noise_mode, unsigned int bound_mode);

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
  double *drift, *noise, *bound, *ic;
  PyArrayObject *_drift, *_noise, *_bound, *_ic;
  PyObject *__drift, *__noise, *__bound, *__ic;
  int nsteps, len_x0;
  int drifttype, noisetype, boundtype;
  if (!PyArg_ParseTuple(args, "OiOiOiOdddi", &__drift, &drifttype, &__noise, &noisetype, &__bound, &boundtype, &__ic, &T_dur, &dt, &dx, &nsteps))
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
  _implicit_time(nsteps, pdfchoice1, pdfchoice2, pdfcurr, drift, noise, bound, ic, len_x0, dt, dx, drifttype, noisetype, boundtype);
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
  double *drift, *noise, *bound, *ic;
  PyArrayObject *_drift, *_noise, *_bound, *_ic;
  PyObject *__drift, *__noise, *__bound, *__ic;
  int nsteps, len_x0;
  int drifttype, noisetype, boundtype;
  if (!PyArg_ParseTuple(args, "OiOiOiOdddi", &__drift, &drifttype, &__noise, &noisetype, &__bound, &boundtype, &__ic, &T_dur, &dt, &dx, &nsteps))
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
  _cn_time(nsteps, pdfchoice1, pdfchoice2, pdfcurr, drift, noise, bound, ic, len_x0, dt, dx, drifttype, noisetype, boundtype);
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

  /* Prepare suminc */
  suminc = (double*)malloc(sizeof(double)*nsteps);
  memset(suminc, 0, nsteps*sizeof(double));

  /* Prepare tmp */
  tmp = (double*)malloc(sizeof(double)*nsteps);
  for (i=1; i<nsteps; i++)
    tmp[i] = -2.*((a1-a2)/(i*tstep)+b1-b2);
  double maxval;
  double toadd;
  double na1a2;
  double coef1, coef2, coef3, coef4;
  checkerr = 0;
  for (n=0; n<nMax; n++) {
    maxval = 0;
    /* V2: na1a2 = n*(a1-a2); */
    na1a2 = n*(a1-a2);
    coef1 = n*(a1+na1a2);
    coef2 = a1+2*na1a2;
    coef3 = (n+1)*(na1a2-a2);
    coef4 = a1-2*a2+2*na1a2;
    for (i=1; i<nsteps; i++) {
      toadd = exp(tmp[i]*coef1)*coef2-
              exp(tmp[i]*coef3)*coef4;
      if (toadd > maxval*suminc[i])
        maxval = toadd/suminc[i];
      suminc[i] += toadd;
      }
    if (maxval < errbnd)
      if (++checkerr >= 3)
        break;
  }
  /*const double sqrtpi = sqrt(2*M_PI); */
  const double oneoversqrtpi = 1/sqrt(2*3.14159265358979);
  /* float itstep; */
  for (i=1; i<nsteps; i++) {
    /* suminc[i] *= exp(-pow((a1+b1*i*tstep),2)/(i*tstep)/2)/sqrtpi/pow(i*tstep, 1.5); */
    /* itstep = i*tstep; */
    suminc[i] *= oneoversqrtpi*exp(-.5*(a1+b1*i*tstep)*(a1+b1*i*tstep)/(i*tstep))*pow(i*tstep, -1.5);
    if (suminc[i] < 0)
      suminc[i] = 0;
  }
  suminc[0] = 0;
  free(tmp);
  return suminc;
}



int _implicit_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr, double* drift, double* noise, double *bound, double *ic, int Xsteps, double dt, double dx, unsigned int drift_mode, unsigned int noise_mode, unsigned int bound_mode) {
  int dmultt=-1, dmultx=-1, nmultt=-1, nmultx=-1, bmultt=-1;
  int i,j,k;
  double dxinv = 1/dx;
  double dtinv = 1/dt;
  double bound_shift, weight_outer, weight_inner;
  int shift_outer, shift_inner;
  double *DU = (double*)malloc((Xsteps-1)*sizeof(double));
  double *D = (double*)malloc(Xsteps*sizeof(double));
  double *DL = (double*)malloc((Xsteps-1)*sizeof(double));
  double *DU_copy = (double*)malloc((Xsteps-1)*sizeof(double));
  double *D_copy = (double*)malloc(Xsteps*sizeof(double));
  double *DL_copy = (double*)malloc((Xsteps-1)*sizeof(double));
  double *pdfcurr_copy = (double*)malloc(Xsteps*sizeof(double));
  memset(pdfchoice1, 0, Tsteps*sizeof(double));
  memset(pdfchoice2, 0, Tsteps*sizeof(double));
  memset(pdfcurr, 0, Xsteps*sizeof(double));
  for (i=0; i<Xsteps; i++) pdfcurr[i] = ic[i];
  /* The different modes define whether we are using a constant
     drift/noise/bound, or one which changes over time, space, or both.  We want
     to store the drift/noise/bound in a single array regardless, so we use
     these to define indexing constants (via the switch statements below) which
     tell us how to access a given element of the array.  This allows us to be
     memory-efficient (passing only a single value if they are constant, but
     still allowing it to vary based on time or space if you are okay with that
     memory usage) without splitting into multiple functions. */
  switch (drift_mode) {
  case 0: /*  Constant drift */
    dmultt = 0; /*  Multiplier for time */
    dmultx = 0; /*  Multiplier for space */
    break;
  case 1: /*  Drift varies over time */
    dmultt = 1;
    dmultx = 0;
    break;
  case 2: /*  Drift varies over space */
    dmultt = 0;
    dmultx = 1;
    break;
  case 3: /*  Drift varies over time and space */
    dmultt = Xsteps;
    dmultx = 1;
    break;
  }
  switch (noise_mode) {
  case 0: /*  Constant noise */
    nmultt = 0; /*  Multiplier for time */
    nmultx = 0; /*  Multiplier for space */
    break;
  case 1: /*  Noise varies over time */
    nmultt = 1;
    nmultx = 0;
    break;
  case 2: /*  Noise varies over space */
    nmultt = 0;
    nmultx = 1;
    break;
  case 3: /*  Noise varies over time and space */
    nmultt = Xsteps;
    nmultx = 1;
    break;
  }
  switch (bound_mode) {
  case 0:
    bmultt = 0;
    break;
  case 1:
    bmultt = 1;
    break;
  }
  double bound_max = bound[0];
  for (i=1; i<Tsteps*bmultt; i++)
    if (bound[i] > bound_max)
      bound_max = bound[i];
  for (i=0; i<Tsteps-1; i++) {
    /*  Make a copy of the current pdf */
    for (j=0; j<Xsteps; j++)
      pdfcurr_copy[j] = pdfcurr[j];
    /*  Compute bound indices */
    bound_shift = bound_max - bound[i*bmultt];
    /* Rounding here is to avoid numerical issues which would cause shift_outer
       to be different from shift_inner, forcing the solver to run twice even
       when we don't have collapsing bounds. */
    shift_outer = (int)floor(1e-10*round(bound_shift*dxinv*1e10));
    shift_inner = (int)ceil(1e-10*round(bound_shift*dxinv*1e10));
    weight_inner = (bound_shift - shift_outer*dx)*dxinv;
    weight_outer = 1 - weight_inner;

    /*  Sum the current pdf and exit if it is small */
    double sumpdfcurr = 0;
    for (j=shift_outer; j<Xsteps-shift_outer; j++)
      sumpdfcurr += pdfcurr[j];
    if (sumpdfcurr < .0001)
      break;


    /* Set up the arguments for lapack.  DL and DU are lower and upper,
       respectively, and D is diagonal. */
    for (j=0; j<Xsteps; j++) {
      D[j]      = 1 + noise[i*nmultt+j*nmultx]*noise[i*nmultt+j*nmultx] * dt * dxinv * dxinv;
      D_copy[j] = D[j];
    }
    for (j=0; j<Xsteps-1; j++) {
      DU[j]      =  .5*drift[i*dmultt+(j+1)*dmultx]*dt * dxinv - .125*(noise[i*nmultt+j*nmultx]+noise[i*nmultt+(j+1)*nmultx])*(noise[i*nmultt+j*nmultx]+noise[i*nmultt+(j+1)*nmultx]) * dt * dxinv * dxinv;
      DL[j]      = -.5*drift[i*dmultt+j*dmultx]*dt * dxinv - .125*(noise[i*nmultt+j*nmultx]+noise[i*nmultt+(j+1)*nmultx])*(noise[i*nmultt+j*nmultx]+noise[i*nmultt+(j+1)*nmultx]) * dt * dxinv * dxinv;
      DU_copy[j] = DU[j];
      DL_copy[j] = DL[j];
    }
    /*  Correction for the implicit method */
    D[shift_outer] += DL[shift_outer];
    DL[shift_outer] = 0;
    D[Xsteps-1-shift_outer] += DU[Xsteps-2-shift_outer];
    DU[Xsteps-2-shift_outer] = 0;
    D_copy[shift_inner] += DL_copy[shift_inner];
    DL_copy[shift_inner] = 0;
    D_copy[Xsteps-1-shift_inner] += DU_copy[Xsteps-2-shift_inner];
    DU_copy[Xsteps-2-shift_inner] = 0;
    if (shift_outer == shift_inner) { /*  Bound falls on a grid here */
      j = Xsteps-shift_outer-1; /*  For convenience */
      for (k=0; k<shift_outer; k++)
        pdfchoice2[i+1] += pdfcurr[k]*weight_outer;
      for (k=Xsteps-shift_outer; k<Xsteps; k++)
        pdfchoice1[i+1] += pdfcurr[k]*weight_outer;
      easy_dgtsv(Xsteps-2*shift_outer, DL+shift_outer, D+shift_outer, DU+shift_outer, pdfcurr+shift_outer);
      pdfchoice1[i+1] += (0.5*dt*dxinv * drift[i*dmultt+j*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+j*nmultx]*noise[i*nmultt+j*nmultx])*pdfcurr[j];
      pdfchoice2[i+1] += (-0.5*dt*dxinv * drift[i*dmultt+shift_outer*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+shift_outer*nmultx]*noise[i*nmultt+shift_outer*nmultx])*pdfcurr[shift_outer];
      for (k=0; k<shift_outer; k++)
        pdfcurr[k] = 0;
      for (k=Xsteps-shift_outer; k<Xsteps; k++)
        pdfcurr[k] = 0;
      /* printf("%f %f    ", pdfcurr[Xsteps-shift_outer-1], pdfcurr[shift_outer]); */
    } else {
      j = Xsteps-shift_outer-1; /*  For convenience */
      for (k=0; k<shift_outer; k++)
        pdfchoice2[i+1] += pdfcurr[k]*weight_outer;
      for (k=Xsteps-shift_outer; k<Xsteps; k++)
        pdfchoice1[i+1] += pdfcurr[k]*weight_outer;
      easy_dgtsv(Xsteps-2*shift_outer, DL+shift_outer, D+shift_outer, DU+shift_outer, pdfcurr+shift_outer);
      pdfchoice1[i+1] += (0.5*dt*dxinv * drift[i*dmultt+j*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+j*nmultx]*noise[i*nmultt+j*nmultx])*pdfcurr[j]*weight_outer;
      pdfchoice2[i+1] += (-0.5*dt*dxinv * drift[i*dmultt+shift_outer*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+shift_outer*nmultx]*noise[i*nmultt+shift_outer*nmultx])*pdfcurr[shift_outer]*weight_outer;
      j = Xsteps-shift_inner-1; /*  For convenience */
      for (k=0; k<shift_inner; k++)
        pdfchoice2[i+1] += pdfcurr_copy[k]*weight_inner;
      for (k=Xsteps-shift_inner; k<Xsteps; k++)
        pdfchoice1[i+1] += pdfcurr_copy[k]*weight_inner;
      easy_dgtsv(Xsteps-2*shift_inner, DL_copy+shift_inner, D_copy+shift_inner, DU_copy+shift_inner, pdfcurr_copy+shift_inner);
      pdfchoice1[i+1] += (0.5*dt*dxinv * drift[i*dmultt+j*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+j*nmultx]*noise[i*nmultt+j*nmultx])*pdfcurr_copy[j]*weight_inner;
      pdfchoice2[i+1] += (-0.5*dt*dxinv * drift[i*dmultt+shift_inner*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+shift_inner*nmultx]*noise[i*nmultt+shift_inner*nmultx])*pdfcurr_copy[shift_inner]*weight_inner;
      for (j=shift_outer; j<Xsteps-shift_outer; j++)
        pdfcurr[j] *= weight_outer;
      for (j=shift_inner; j<Xsteps-shift_inner; j++)
        pdfcurr[j] += pdfcurr_copy[j]*weight_inner;
      for (k=0; k<shift_outer; k++)
        pdfcurr[k] = 0;
      for (k=Xsteps-shift_outer; k<Xsteps; k++)
        pdfcurr[k] = 0;
      for (k=0; k<shift_inner; k++)
        pdfcurr_copy[k] = 0;
      for (k=Xsteps-shift_inner; k<Xsteps; k++)
        pdfcurr_copy[k] = 0;
    }

  }
  /*  Normalize pdfchoice1 and pdfchoice2 */
  double sumpdfs = 0;
  for (i=0; i<Tsteps; i++)
    sumpdfs += pdfchoice1[i] + pdfchoice2[i];
  if (sumpdfs > 1) {
    for (i=0; i<Tsteps; i++) {
      pdfchoice1[i] /= sumpdfs;
      pdfchoice2[i] /= sumpdfs;
    }
  }
  /* Scale the output to make it a pdf, not a pmf */
  for (i=0; i<Tsteps; i++) {
    pdfchoice1[i] *= dtinv;
    pdfchoice2[i] *= dtinv;
  }
  /* Free memory */
  free(pdfcurr_copy);
  free(D);
  free(DU);
  free(DL);
  free(D_copy);
  free(DU_copy);
  free(DL_copy);
  return 0;
}


int _cn_time(int Tsteps, double *pdfchoice1, double *pdfchoice2, double *pdfcurr, double* drift, double* noise, double *bound, double *ic, int Xsteps, double dt, double dx, unsigned int drift_mode, unsigned int noise_mode, unsigned int bound_mode) {
  int dmultt=-1, dmultx=-1, nmultt=-1, nmultx=-1, bmultt=-1;
  int i,j,k;
  double dxinv = 1/dx;
  double bound_shift, weight_outer = 1, weight_inner = 0;
  double weight_outer_prev = 1, weight_inner_prev = 0;
  int x_index_outer = 0, x_index_inner = 0;
  int x_index_outer_prev = 0, x_index_inner_prev = 0;
  int x_index_outer_shift, x_index_inner_shift;
  int n_outer = Xsteps, n_inner = Xsteps, n_outer_prev = Xsteps, n_inner_prev = Xsteps;
  int n_solve, start_solve, start_prev, start_current;
  double local_dt;
  double flux_outer_B_choice_upper, flux_inner_B_choice_upper;
  double flux_outer_B_choice_lower, flux_inner_B_choice_lower;
  double sumpdfcurr;

  double *DU = (double*)malloc((Xsteps-1)*sizeof(double));
  double *D = (double*)malloc(Xsteps*sizeof(double));
  double *DL = (double*)malloc((Xsteps-1)*sizeof(double));
  double *DU_prev = (double*)malloc((Xsteps-1)*sizeof(double));
  double *D_prev = (double*)malloc(Xsteps*sizeof(double));
  double *DL_prev = (double*)malloc((Xsteps-1)*sizeof(double));
  double *pdf_outer = (double*)malloc(Xsteps*sizeof(double));
  double *pdf_inner = (double*)malloc(Xsteps*sizeof(double));
  double *pdf_outer_prev = (double*)malloc(Xsteps*sizeof(double));
  double *pdf_inner_prev = (double*)malloc(Xsteps*sizeof(double));
  double *rhs_full = (double*)malloc(Xsteps*sizeof(double));
  double *rhs = (double*)malloc(Xsteps*sizeof(double));
  double *pdfchoice1_tmp = (double*)malloc((Tsteps+1)*sizeof(double));
  double *pdfchoice2_tmp = (double*)malloc((Tsteps+1)*sizeof(double));

  memset(pdfchoice1, 0, Tsteps*sizeof(double));
  memset(pdfchoice2, 0, Tsteps*sizeof(double));
  memset(pdfchoice1_tmp, 0, (Tsteps+1)*sizeof(double));
  memset(pdfchoice2_tmp, 0, (Tsteps+1)*sizeof(double));
  memset(pdfcurr, 0, Xsteps*sizeof(double));
  for (i=0; i<Xsteps; i++) {
    pdfcurr[i] = ic[i];
    pdf_outer[i] = ic[i];
    pdf_inner[i] = ic[i];
  }

  switch (drift_mode) {
  case 0:
    dmultt = 0;
    dmultx = 0;
    break;
  case 1:
    dmultt = 1;
    dmultx = 0;
    break;
  case 2:
    dmultt = 0;
    dmultx = 1;
    break;
  case 3:
    dmultt = Xsteps;
    dmultx = 1;
    break;
  }
  switch (noise_mode) {
  case 0:
    nmultt = 0;
    nmultx = 0;
    break;
  case 1:
    nmultt = 1;
    nmultx = 0;
    break;
  case 2:
    nmultt = 0;
    nmultx = 1;
    break;
  case 3:
    nmultt = Xsteps;
    nmultx = 1;
    break;
  }
  switch (bound_mode) {
  case 0:
    bmultt = 0;
    break;
  case 1:
    bmultt = 1;
    break;
  }

  for (i=0; i<Tsteps; i++) {
    for (j=0; j<n_outer; j++)
      pdf_outer_prev[j] = pdf_outer[j];
    for (j=0; j<n_inner; j++)
      pdf_inner_prev[j] = pdf_inner[j];
    n_outer_prev = n_outer;
    n_inner_prev = n_inner;

    sumpdfcurr = 0;
    for (j=0; j<Xsteps; j++)
      sumpdfcurr += pdfcurr[j];
    if (sumpdfcurr <= .0001)
      break;

    bound_shift = bound[0] - bound[i*bmultt];
    x_index_outer_prev = x_index_outer;
    x_index_inner_prev = x_index_inner;
    x_index_outer = (int)floor(bound_shift*dxinv);
    x_index_inner = (int)ceil(bound_shift*dxinv);
    x_index_outer_shift = x_index_outer - x_index_outer_prev;
    x_index_inner_shift = x_index_inner - x_index_inner_prev;
    weight_inner_prev = weight_inner;
    weight_outer_prev = weight_outer;
    weight_inner = (bound_shift - x_index_outer*dx)*dxinv;
    weight_outer = 1 - weight_inner;
    n_outer = Xsteps - 2*x_index_outer;
    n_inner = Xsteps - 2*x_index_inner;
    local_dt = dt;

    if (n_outer <= 0)
      break;

    /* Current outer matrix: I + .5*M(t). */
    start_current = x_index_outer;
    for (j=0; j<n_outer; j++) {
      k = start_current+j;
      D[j] = 1 + .5*noise[i*nmultt+k*nmultx]*noise[i*nmultt+k*nmultx] * local_dt * dxinv * dxinv;
    }
    for (j=0; j<n_outer-1; j++) {
      k = start_current+j;
      DU[j] = .25*drift[i*dmultt+(k+1)*dmultx]*local_dt * dxinv - .0625*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx])*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
      DL[j] = -.25*drift[i*dmultt+k*dmultx]*local_dt * dxinv - .0625*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx])*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
    }
    if (n_outer > 1) {
      D[0] += DL[0];
      DL[0] = 0;
      D[n_outer-1] += DU[n_outer-2];
      DU[n_outer-2] = 0;
    }

    /* Previous outer matrix: I - .5*M(prev_t). */
    start_prev = x_index_outer_prev;
    int iprev = i == 0 ? 0 : i-1;
    for (j=0; j<n_outer_prev; j++) {
      k = start_prev+j;
      D_prev[j] = 1 - .5*noise[iprev*nmultt+k*nmultx]*noise[iprev*nmultt+k*nmultx] * local_dt * dxinv * dxinv;
    }
    for (j=0; j<n_outer_prev-1; j++) {
      k = start_prev+j;
      DU_prev[j] = -.25*drift[iprev*dmultt+(k+1)*dmultx]*local_dt * dxinv + .0625*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx])*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
      DL_prev[j] = .25*drift[iprev*dmultt+k*dmultx]*local_dt * dxinv + .0625*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx])*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
    }
    if (n_outer_prev > 1) {
      D_prev[0] += DL_prev[0];
      DL_prev[0] = 0;
      D_prev[n_outer_prev-1] += DU_prev[n_outer_prev-2];
      DU_prev[n_outer_prev-2] = 0;
    }
    for (j=0; j<n_outer_prev; j++) {
      rhs_full[j] = D_prev[j]*pdf_outer_prev[j];
      if (j > 0)
        rhs_full[j] += DL_prev[j-1]*pdf_outer_prev[j-1];
      if (j < n_outer_prev-1)
        rhs_full[j] += DU_prev[j]*pdf_outer_prev[j+1];
    }
    for (j=0; j<n_outer; j++)
      rhs[j] = rhs_full[x_index_outer_shift+j];
    easy_dgtsv(n_outer, DL, D, DU, rhs);
    for (j=0; j<n_outer; j++)
      pdf_outer[j] = rhs[j];

    if (x_index_inner == x_index_outer) {
      for (j=0; j<n_outer; j++)
        pdf_inner[j] = pdf_outer[j];
      n_inner = n_outer;
    } else if (n_inner > 0) {
      /* Current inner matrix: I + .5*M(t). */
      n_solve = n_inner;
      start_solve = x_index_inner;
      for (j=0; j<n_solve; j++) {
        k = start_solve+j;
        D[j] = 1 + .5*noise[i*nmultt+k*nmultx]*noise[i*nmultt+k*nmultx] * local_dt * dxinv * dxinv;
      }
      for (j=0; j<n_solve-1; j++) {
        k = start_solve+j;
        DU[j] = .25*drift[i*dmultt+(k+1)*dmultx]*local_dt * dxinv - .0625*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx])*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
        DL[j] = -.25*drift[i*dmultt+k*dmultx]*local_dt * dxinv - .0625*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx])*(noise[i*nmultt+k*nmultx]+noise[i*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
      }
      if (n_solve > 1) {
        D[0] += DL[0];
        DL[0] = 0;
        D[n_solve-1] += DU[n_solve-2];
        DU[n_solve-2] = 0;
      }

      /* Previous inner matrix: I - .5*M(prev_t). */
      start_prev = x_index_inner_prev;
      for (j=0; j<n_inner_prev; j++) {
        k = start_prev+j;
        D_prev[j] = 1 - .5*noise[iprev*nmultt+k*nmultx]*noise[iprev*nmultt+k*nmultx] * local_dt * dxinv * dxinv;
      }
      for (j=0; j<n_inner_prev-1; j++) {
        k = start_prev+j;
        DU_prev[j] = -.25*drift[iprev*dmultt+(k+1)*dmultx]*local_dt * dxinv + .0625*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx])*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
        DL_prev[j] = .25*drift[iprev*dmultt+k*dmultx]*local_dt * dxinv + .0625*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx])*(noise[iprev*nmultt+k*nmultx]+noise[iprev*nmultt+(k+1)*nmultx]) * local_dt * dxinv * dxinv;
      }
      if (n_inner_prev > 1) {
        D_prev[0] += DL_prev[0];
        DL_prev[0] = 0;
        D_prev[n_inner_prev-1] += DU_prev[n_inner_prev-2];
        DU_prev[n_inner_prev-2] = 0;
      }
      for (j=0; j<n_inner_prev; j++) {
        rhs_full[j] = D_prev[j]*pdf_inner_prev[j];
        if (j > 0)
          rhs_full[j] += DL_prev[j-1]*pdf_inner_prev[j-1];
        if (j < n_inner_prev-1)
          rhs_full[j] += DU_prev[j]*pdf_inner_prev[j+1];
      }
      for (j=0; j<n_solve; j++)
        rhs[j] = rhs_full[x_index_inner_shift+j];
      easy_dgtsv(n_solve, DL, D, DU, rhs);
      for (j=0; j<n_solve; j++)
        pdf_inner[j] = rhs[j];
    }

    for (k=0; k<x_index_outer_shift; k++)
      pdfchoice2_tmp[i+1] += pdf_outer_prev[k]*weight_outer_prev;
    for (k=n_outer_prev-x_index_outer_shift; k<n_outer_prev; k++)
      pdfchoice1_tmp[i+1] += pdf_outer_prev[k]*weight_outer_prev;
    for (k=0; k<x_index_inner_shift; k++)
      pdfchoice2_tmp[i+1] += pdf_inner_prev[k]*weight_inner_prev;
    for (k=n_inner_prev-x_index_inner_shift; k<n_inner_prev; k++)
      pdfchoice1_tmp[i+1] += pdf_inner_prev[k]*weight_inner_prev;

    for (j=0; j<Xsteps; j++)
      pdfcurr[j] = 0;
    for (j=0; j<n_outer; j++)
      pdfcurr[x_index_outer+j] += weight_outer*pdf_outer[j];
    if (n_inner > 0)
      for (j=0; j<n_inner; j++)
        pdfcurr[x_index_inner+j] += weight_inner*pdf_inner[j];

    int outer_upper = Xsteps-1-x_index_outer;
    int outer_lower = x_index_outer;
    int inner_upper = Xsteps-1-x_index_inner;
    int inner_lower = x_index_inner;
    flux_outer_B_choice_upper = 0.5*dt*dxinv * drift[i*dmultt+outer_upper*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+outer_upper*nmultx]*noise[i*nmultt+outer_upper*nmultx];
    flux_outer_B_choice_lower = -0.5*dt*dxinv * drift[i*dmultt+outer_lower*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+outer_lower*nmultx]*noise[i*nmultt+outer_lower*nmultx];
    flux_inner_B_choice_upper = 0.5*dt*dxinv * drift[i*dmultt+inner_upper*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+inner_upper*nmultx]*noise[i*nmultt+inner_upper*nmultx];
    flux_inner_B_choice_lower = -0.5*dt*dxinv * drift[i*dmultt+inner_lower*dmultx] + 0.5*dt*dxinv*dxinv * noise[i*nmultt+inner_lower*nmultx]*noise[i*nmultt+inner_lower*nmultx];

    pdfchoice1_tmp[i+1] += 0.5*weight_outer * pdf_outer[n_outer-1] * flux_outer_B_choice_upper;
    pdfchoice2_tmp[i+1] += 0.5*weight_outer * pdf_outer[0] * flux_outer_B_choice_lower;
    pdfchoice1_tmp[i]   += 0.5*weight_outer * pdf_outer[n_outer-1] * flux_outer_B_choice_upper;
    pdfchoice2_tmp[i]   += 0.5*weight_outer * pdf_outer[0] * flux_outer_B_choice_lower;
    if (n_inner > 0) {
      pdfchoice1_tmp[i+1] += 0.5*weight_inner * pdf_inner[n_inner-1] * flux_inner_B_choice_upper;
      pdfchoice2_tmp[i+1] += 0.5*weight_inner * pdf_inner[0] * flux_inner_B_choice_lower;
      pdfchoice1_tmp[i]   += 0.5*weight_inner * pdf_inner[n_inner-1] * flux_inner_B_choice_upper;
      pdfchoice2_tmp[i]   += 0.5*weight_inner * pdf_inner[0] * flux_inner_B_choice_lower;
    }

    if (bound[i*bmultt] < dx) {
      pdfchoice1_tmp[i+1] *= (1 + (1 - bound[i*bmultt]*dxinv));
      pdfchoice2_tmp[i+1] *= (1 + (1 - bound[i*bmultt]*dxinv));
    }
  }

  pdfchoice1[0] = 0;
  pdfchoice2[0] = 0;
  for (i=1; i<Tsteps; i++) {
    pdfchoice1[i] = .5*(pdfchoice1_tmp[i] + pdfchoice1_tmp[i-1]);
    pdfchoice2[i] = .5*(pdfchoice2_tmp[i] + pdfchoice2_tmp[i-1]);
  }

  free(pdfchoice1_tmp);
  free(pdfchoice2_tmp);
  free(rhs);
  free(rhs_full);
  free(pdf_inner_prev);
  free(pdf_outer_prev);
  free(pdf_inner);
  free(pdf_outer);
  free(D);
  free(DU);
  free(DL);
  free(D_prev);
  free(DU_prev);
  free(DL_prev);
  return 0;
}
