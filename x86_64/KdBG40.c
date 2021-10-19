/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__KdBG
#define _nrn_initial _nrn_initial__KdBG
#define nrn_cur _nrn_cur__KdBG
#define _nrn_current _nrn_current__KdBG
#define nrn_jacob _nrn_jacob__KdBG
#define nrn_state _nrn_state__KdBG
#define _net_receive _net_receive__KdBG 
#define rates rates__KdBG 
#define states states__KdBG 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gbar _p[0]
#define ik _p[1]
#define xs _p[2]
#define ys _p[3]
#define Dxs _p[4]
#define Dys _p[5]
#define _g _p[6]
#define _ion_ik	*_ppvar[0]._pval
#define _ion_dikdv	*_ppvar[1]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_KdBG", _hoc_setdata,
 "rates_KdBG", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define FRT FRT_KdBG
 double FRT = 39;
#define Ky Ky_KdBG
 double Ky = 0.001;
#define Kx Kx_KdBG
 double Kx = 1;
#define q10 q10_KdBG
 double q10 = 1;
#define tauy tauy_KdBG
 double tauy = 100;
#define taux taux_KdBG
 double taux = 1;
#define vhalfy vhalfy_KdBG
 double vhalfy = -90;
#define vhalfx vhalfx_KdBG
 double vhalfx = -48;
#define xinf xinf_KdBG
 double xinf = 0;
#define xtau xtau_KdBG
 double xtau = 0;
#define yinf yinf_KdBG
 double yinf = 0;
#define ytau ytau_KdBG
 double ytau = 0;
#define zettay zettay_KdBG
 double zettay = -1.5;
#define zettax zettax_KdBG
 double zettax = 2.5;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Kx_KdBG", "1/ms",
 "Ky_KdBG", "1/ms",
 "zettax_KdBG", "1",
 "zettay_KdBG", "1",
 "vhalfx_KdBG", "mV",
 "vhalfy_KdBG", "mV",
 "taux_KdBG", "ms",
 "tauy_KdBG", "ms",
 "q10_KdBG", "1",
 "FRT_KdBG", "coulombs/joule",
 "xtau_KdBG", "ms",
 "ytau_KdBG", "ms",
 "xinf_KdBG", "1",
 "yinf_KdBG", "1",
 "gbar_KdBG", "S/cm2",
 "ik_KdBG", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double v = 0;
 static double xs0 = 0;
 static double ys0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Kx_KdBG", &Kx_KdBG,
 "Ky_KdBG", &Ky_KdBG,
 "zettax_KdBG", &zettax_KdBG,
 "zettay_KdBG", &zettay_KdBG,
 "vhalfx_KdBG", &vhalfx_KdBG,
 "vhalfy_KdBG", &vhalfy_KdBG,
 "taux_KdBG", &taux_KdBG,
 "tauy_KdBG", &tauy_KdBG,
 "q10_KdBG", &q10_KdBG,
 "FRT_KdBG", &FRT_KdBG,
 "xtau_KdBG", &xtau_KdBG,
 "ytau_KdBG", &ytau_KdBG,
 "xinf_KdBG", &xinf_KdBG,
 "yinf_KdBG", &yinf_KdBG,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"KdBG",
 "gbar_KdBG",
 0,
 "ik_KdBG",
 0,
 "xs_KdBG",
 "ys_KdBG",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 7, _prop);
 	/*initialize range parameters*/
 	gbar = 0.001;
 	_prop->param = _p;
 	_prop->param_size = 7;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[1]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _KdBG40_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 7, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 KdBG /home/dhh/soltesz-lab/qan-hfo/v2-7272021/x86_64/KdBG40.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96480.0;
 static double R = 8.314;
static int _reset;
static char *modelname = "Kd current";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dxs = ( xinf - xs ) / xtau ;
   Dys = ( yinf - ys ) / ytau ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dxs = Dxs  / (1. - dt*( ( ( ( - 1.0 ) ) ) / xtau )) ;
 Dys = Dys  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ytau )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    xs = xs + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / xtau)))*(- ( ( ( xinf ) ) / xtau ) / ( ( ( ( - 1.0 ) ) ) / xtau ) - xs) ;
    ys = ys + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ytau)))*(- ( ( ( yinf ) ) / ytau ) / ( ( ( ( - 1.0 ) ) ) / ytau ) - ys) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   double _la , _lb , _lT , _lqt ;
 _lT = celsius + 273.15 ;
   _lqt = pow( q10 , ( ( celsius - 35.0 ) / 10.0 ) ) ;
   _la = _lqt * Kx * exp ( ( 1.0e-3 ) * zettax * ( _lv - vhalfx ) * FRT ) ;
   _lb = _lqt * Kx * exp ( ( 1.0e-3 ) * - zettax * ( _lv - vhalfx ) * FRT ) ;
   xinf = _la / ( _la + _lb ) ;
   xtau = 1.0 / ( _la + _lb ) + taux ;
   _la = _lqt * Ky * exp ( ( 1.0e-3 ) * zettay * ( _lv - vhalfy ) * FRT ) ;
   _lb = _lqt * Ky * exp ( ( 1.0e-3 ) * - zettay * ( _lv - vhalfy ) * FRT ) ;
   yinf = _la / ( _la + _lb ) ;
   ytau = 1.0 / ( _la + _lb ) + tauy ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  xs = xs0;
  ys = ys0;
 {
   rates ( _threadargscomma_ v ) ;
   xs = xinf ;
   ys = yinf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   ik = gbar * xs * ys * ( v + 90.0 ) ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 { error =  states();
 if(error){fprintf(stderr,"at line 58 in file KdBG40.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(xs) - _p;  _dlist1[0] = &(Dxs) - _p;
 _slist1[1] = &(ys) - _p;  _dlist1[1] = &(Dys) - _p;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/home/dhh/soltesz-lab/qan-hfo/v2-7272021/KdBG40.mod";
static const char* nmodl_file_text = 
  "TITLE Kd current\n"
  "\n"
  "COMMENT \n"
  "		For act & inact tau, \n"
  "		  Storm JF (1988) Nature, 336:379-381\n"
  "		For vhalf and slope of xinf and yinf,\n"
  "		  BossuGahwiler1_JP96.pdf &  TytgatDaenens_BJP97(kv11).pdf \n"
  "		To fit simulation to the CA3 data, LSH changed\n"
  "			tauy, 150 -> 100 ms (Hyun et al JP 2013)\n"
  "			vhalfx, -55 -> -48(Saviane JP 2003 and Hyun JP 2013)\n"
  "			vhalfy, -88 -> -90 (Saviane JP 2003)\n"
  "			Ky, 0.6e-3 -> 1e-3\n"
  "			zettax, 2 -> 2.5 (Hyun JP 2013)\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX KdBG\n"
  "	USEION k WRITE ik\n"
  "	RANGE  gbar,ik\n"
  "	GLOBAL xtau, ytau, xinf, yinf\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(S)	= (siemens)\n"
  "	(mA)	= (milliamp)\n"
  "	(mV)	= (millivolt)\n"
  "	FARADAY	= 96480 (coulombs)\n"
  "	R	= 8.314  (joule/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v		(mV)\n"
  "	gbar	= 1.0e-3	(S/cm2)\n"
  "	celsius	= 25	(degC)\n"
  "	Kx = 1   (1/ms)\n"
  "	Ky	=   1e-3	(1/ms)\n"
  "	zettax	=  2.5		(1)\n"
  "	zettay	=  -1.5		(1)\n"
  "	vhalfx	= -48.0		(mV)\n"
  "	vhalfy	= -90.0		(mV)\n"
  "	taux	=   1		(ms)\n"
  "	tauy	=   100		(ms)\n"
  "	q10	= 1.0	(1)    : no temp dependence\n"
  "	FRT = 39 (coulombs/joule) \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ik     	(mA/cm2)\n"
  "	xtau    (ms)\n"
  "	ytau    (ms)\n"
  "	xinf	(1)\n"
  "	yinf	(1)\n"
  "}\n"
  "\n"
  "STATE { xs ys }\n"
  "\n"
  "BREAKPOINT { \n"
  "	SOLVE states METHOD cnexp\n"
  "	ik= gbar * xs * ys * ( v + 90.0 ) \n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "	rates(v)\n"
  "	xs'= (xinf- xs)/ xtau	\n"
  "	ys'= (yinf- ys)/ ytau\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	xs= xinf\n"
  "	ys= yinf\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) { LOCAL a, b, T, qt\n"
  "	T = celsius + 273.15  \n"
  "	qt = q10 ^( (celsius-35.0) / 10.0(K) )\n"
  "	a = qt*Kx*exp( (1.0e-3)*  zettax*(v-vhalfx)*FRT )\n"
  "	b = qt*Kx*exp( (1.0e-3)* -zettax*(v-vhalfx)*FRT )\n"
  "	xinf = a / ( a + b )\n"
  "	xtau = 1 /(a + b)+ taux\n"
  "\n"
  "	a = qt*Ky*exp( (1.0e-3)*  zettay* (v-vhalfy)*FRT )\n"
  "	b = qt*Ky*exp( (1.0e-3)* -zettay* (v-vhalfy)*FRT )\n"
  "	yinf = a   / ( a + b )\n"
  "	ytau = 1.0 / ( a + b ) + tauy\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
