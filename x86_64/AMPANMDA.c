/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__AMPANMDA
#define _nrn_initial _nrn_initial__AMPANMDA
#define nrn_cur _nrn_cur__AMPANMDA
#define _nrn_current _nrn_current__AMPANMDA
#define nrn_jacob _nrn_jacob__AMPANMDA
#define nrn_state _nrn_state__AMPANMDA
#define _net_receive _net_receive__AMPANMDA 
#define state state__AMPANMDA 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gid _p[0]
#define e _p[1]
#define tau1ampa _p[2]
#define tau2ampa _p[3]
#define tau1nmda _p[4]
#define tau2nmda _p[5]
#define eta _p[6]
#define gamma _p[7]
#define Mg _p[8]
#define thresh _p[9]
#define d _p[10]
#define p _p[11]
#define dM _p[12]
#define dV _p[13]
#define ptau _p[14]
#define wmax _p[15]
#define wmin _p[16]
#define pon _p[17]
#define i _p[18]
#define wfampa _p[19]
#define wfnmda _p[20]
#define Aampa _p[21]
#define Bampa _p[22]
#define Anmda _p[23]
#define Bnmda _p[24]
#define tpost _p[25]
#define factorampa _p[26]
#define factornmda _p[27]
#define DAampa _p[28]
#define DBampa _p[29]
#define DAnmda _p[30]
#define DBnmda _p[31]
#define v _p[32]
#define _g _p[33]
#define _tsav _p[34]
#define _nd_area  *_ppvar[0]._pval
 
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_Mgblock();
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

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "Mgblock", _hoc_Mgblock,
 0, 0
};
#define Mgblock Mgblock_AMPANMDA
 extern double Mgblock( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define pi pi_AMPANMDA
 double pi = 3.14159;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "e", "mV",
 "tau1ampa", "ms",
 "tau2ampa", "ms",
 "tau1nmda", "ms",
 "tau2nmda", "ms",
 "eta", "/mM",
 "gamma", "/mV",
 "Mg", "mM",
 "thresh", "mV",
 "dM", "ms",
 "dV", "ms",
 "ptau", "ms",
 "Aampa", "uS",
 "Bampa", "uS",
 "Anmda", "uS",
 "Bnmda", "uS",
 "i", "nA",
 0,0
};
 static double Anmda0 = 0;
 static double Aampa0 = 0;
 static double Bnmda0 = 0;
 static double Bampa0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "pi_AMPANMDA", &pi_AMPANMDA,
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
 
#define _watch_array _ppvar + 3 
 
#define _fnc_index 5
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   Prop* _prop = ((Point_process*)_vptr)->_prop;
   if (_prop) { _nrn_free_watch(_prop->dparam, 3, 2);}
   if (_prop) { _nrn_free_fornetcon(&(_prop->dparam[_fnc_index]._pvoid));}
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"AMPANMDA",
 "gid",
 "e",
 "tau1ampa",
 "tau2ampa",
 "tau1nmda",
 "tau2nmda",
 "eta",
 "gamma",
 "Mg",
 "thresh",
 "d",
 "p",
 "dM",
 "dV",
 "ptau",
 "wmax",
 "wmin",
 "pon",
 0,
 "i",
 "wfampa",
 "wfnmda",
 0,
 "Aampa",
 "Bampa",
 "Anmda",
 "Bnmda",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 35, _prop);
 	/*initialize range parameters*/
 	gid = 0;
 	e = 0;
 	tau1ampa = 0.1;
 	tau2ampa = 10;
 	tau1nmda = 0.6;
 	tau2nmda = 55;
 	eta = 0.33;
 	gamma = 0.14;
 	Mg = 1;
 	thresh = 0;
 	d = 8;
 	p = 1.2;
 	dM = -22;
 	dV = 5;
 	ptau = 10;
 	wmax = 0.005;
 	wmin = 1e-05;
 	pon = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 35;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern int _nrn_netcon_args(void*, double***);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _AMPANMDA_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 35, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "watch");
  hoc_register_dparam_semantics(_mechtype, 4, "watch");
  hoc_register_dparam_semantics(_mechtype, 5, "fornetcon");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 4;
 add_nrn_fornetcons(_mechtype, _fnc_index);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 AMPANMDA /home/dhh/soltesz-lab/qan-hfo/v2-7272021/x86_64/AMPANMDA.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Dual-exponential Exp2Syn with added dual-exponential model of NMDA receptors";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   DAampa = - Aampa / tau1ampa ;
   DBampa = - Bampa / tau2ampa ;
   DAnmda = - Anmda / tau1nmda ;
   DBnmda = - Bnmda / tau2nmda ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 DAampa = DAampa  / (1. - dt*( ( - 1.0 ) / tau1ampa )) ;
 DBampa = DBampa  / (1. - dt*( ( - 1.0 ) / tau2ampa )) ;
 DAnmda = DAnmda  / (1. - dt*( ( - 1.0 ) / tau1nmda )) ;
 DBnmda = DBnmda  / (1. - dt*( ( - 1.0 ) / tau2nmda )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
    Aampa = Aampa + (1. - exp(dt*(( - 1.0 ) / tau1ampa)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau1ampa ) - Aampa) ;
    Bampa = Bampa + (1. - exp(dt*(( - 1.0 ) / tau2ampa)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau2ampa ) - Bampa) ;
    Anmda = Anmda + (1. - exp(dt*(( - 1.0 ) / tau1nmda)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau1nmda ) - Anmda) ;
    Bnmda = Bnmda + (1. - exp(dt*(( - 1.0 ) / tau2nmda)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau2nmda ) - Bnmda) ;
   }
  return 0;
}
 
static double _watch1_cond(_pnt) Point_process* _pnt; {
 	double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
	_thread= (Datum*)0; _nt = (_NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( thresh ) ;
}
 
static void _net_receive (_pnt, _args, _lflag) Point_process* _pnt; double* _args; double _lflag; 
{  double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   int _watch_rm = 0;
   _thread = (Datum*)0; _nt = (_NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 0.0 ) {
     if ( pon  == 1.0 ) {
       wfampa = ( _args[0] + _args[2] ) * factorampa ;
       _args[3] = t ;
       _args[2] = _args[2] * ( 1.0 - ( d * exp ( - pow( ( ( tpost - t ) - dM ) , 2.0 ) / ( 2.0 * dV * dV ) ) ) / ( sqrt ( 2.0 * pi ) * dV ) ) ;
       }
     else {
       wfampa = _args[0] * factorampa ;
       }
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Aampa;
    double __primary = (Aampa + wfampa) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau1ampa ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau1ampa ) - __primary );
    Aampa += __primary;
  } else {
 Aampa = Aampa + wfampa ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Bampa;
    double __primary = (Bampa + wfampa) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau2ampa ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau2ampa ) - __primary );
    Bampa += __primary;
  } else {
 Bampa = Bampa + wfampa ;
       }
 wfnmda = _args[1] * factornmda ;
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Anmda;
    double __primary = (Anmda + wfnmda) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau1nmda ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau1nmda ) - __primary );
    Anmda += __primary;
  } else {
 Anmda = Anmda + wfnmda ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Bnmda;
    double __primary = (Bnmda + wfnmda) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau2nmda ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau2nmda ) - __primary );
    Bnmda += __primary;
  } else {
 Bnmda = Bnmda + wfnmda ;
       }
 }
   else if ( _lflag  == 1.0 ) {
       _nrn_watch_activate(_watch_array, _watch1_cond, 1, _pnt, _watch_rm++, 2.0);
 }
   else if ( _lflag  == 2.0 ) {
     if ( pon  == 1.0 ) {
       tpost = t ;
       {int _ifn1, _nfn1; double* _fnargs1, **_fnargslist1;
	_nfn1 = _nrn_netcon_args(_ppvar[_fnc_index]._pvoid, &_fnargslist1);
	for (_ifn1 = 0; _ifn1 < _nfn1; ++_ifn1) {
 	 _fnargs1 = _fnargslist1[_ifn1];
 {
         _fnargs1[2] = _fnargs1[2] + ( wmax - _fnargs1[0] - _fnargs1[2] ) * p * exp ( ( _fnargs1[3] - t ) / ptau ) ;
         }
       	}}
 }
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    _NrnThread* _nt = (_NrnThread*)_pnt->_vnt;
 _args[1] = _args[1] ;
   _args[2] = 0.0 ;
   _args[3] = - 1e9 ;
   }
 
double Mgblock ( _threadargsprotocomma_ double _lv ) {
   double _lMgblock;
 _lMgblock = 1.0 / ( 1.0 + eta * Mg * exp ( - ( gamma * _lv ) ) ) ;
   
return _lMgblock;
 }
 
static double _hoc_Mgblock(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (_NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  Mgblock ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ return 4;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 4; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  Anmda = Anmda0;
  Aampa = Aampa0;
  Bnmda = Bnmda0;
  Bampa = Bampa0;
 {
   double _ltpampa , _ltpnmda ;
 if ( tau1ampa / tau2ampa > .9999 ) {
     tau1ampa = .9999 * tau2ampa ;
     }
   Aampa = 0.0 ;
   Bampa = 0.0 ;
   _ltpampa = ( tau1ampa * tau2ampa ) / ( tau2ampa - tau1ampa ) * log ( tau2ampa / tau1ampa ) ;
   factorampa = - exp ( - _ltpampa / tau1ampa ) + exp ( - _ltpampa / tau2ampa ) ;
   factorampa = 1.0 / factorampa ;
   if ( tau1nmda / tau2nmda > .9999 ) {
     tau1nmda = 0.9999 * tau2nmda ;
     }
   Anmda = 0.0 ;
   Bnmda = 0.0 ;
   _ltpnmda = ( tau1nmda * tau2nmda ) / ( tau2nmda - tau1nmda ) * log ( tau2nmda / tau1nmda ) ;
   factornmda = - exp ( - _ltpnmda / tau1nmda ) + exp ( - _ltpnmda / tau2nmda ) ;
   factornmda = 1.0 / factornmda ;
   Mgblock ( _threadargscomma_ v ) ;
   tpost = - 1e9 ;
   net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  0.0 , 1.0 ) ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   i = ( Bampa - Aampa ) * ( v - e ) + ( Bnmda - Anmda ) * Mgblock ( _threadargscomma_ v ) * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 {   state(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(Aampa) - _p;  _dlist1[0] = &(DAampa) - _p;
 _slist1[1] = &(Bampa) - _p;  _dlist1[1] = &(DBampa) - _p;
 _slist1[2] = &(Anmda) - _p;  _dlist1[2] = &(DAnmda) - _p;
 _slist1[3] = &(Bnmda) - _p;  _dlist1[3] = &(DBnmda) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/dhh/soltesz-lab/qan-hfo/v2-7272021/AMPANMDA.mod";
static const char* nmodl_file_text = 
  "TITLE Dual-exponential Exp2Syn with added dual-exponential model of NMDA receptors\n"
  "\n"
  "COMMENT\n"
  "Written by Darian Hadjiabadi\n"
  "AMPA rececptor adapted from h.Exp2Syn\n"
  "NMDA receptor Adapted from \"Contribution of NMDA Receptor Channels to the Expression of LTP in the\n"
  "Hippocampal Dentate Gyrus by Zhuo Wang,\" Dong Song, and Theodore W. Berger\n"
  "Hippocampus, vol. 12, pp. 680-688, 2002.\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "    POINT_PROCESS AMPANMDA\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE gid, tau1ampa, tau2ampa, tau1nmda, tau2nmda, e, i, Mg, eta, gamma, wfampa, wfnmda, thresh\n"
  "    RANGE d, p, dM, dV, ptau, wmax, wmin, pon\n"
  "    THREADSAFE\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (nA) = (nanoamp)\n"
  "    (mV) = (millivolt)\n"
  "    (uS) = (microsiemens)\n"
  "    (mM) = (milli/liter)\n"
  "    (S)  = (siemens)\n"
  "    (pS) = (picosiemens)\n"
  "    (um) = (micron)\n"
  "    (J)  = (joules)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    gid = 0\n"
  "    e = 0 (mV)\n"
  ": Parameters control AMPA gating\n"
  "    tau1ampa = 0.1 (ms)\n"
  "    tau2ampa = 10.0 (ms)\n"
  "    \n"
  ": Parameters Control Neurotransmitter and Voltage-dependent gating of NMDAR\n"
  "    tau1nmda = 0.6  (ms)\n"
  "    tau2nmda = 55  (ms)\n"
  ": Parameters Control voltage-dependent gating of NMDAR\n"
  "    eta = 0.33 (/mM)\n"
  "    gamma = 0.14 (/mV)\n"
  ": Parameters Control Mg block of NMDAR\n"
  "    Mg = 1  (mM)\n"
  ": Spike threshold for calculating plasticity changes\n"
  "    thresh = 0 (mV)\n"
  ": plasticity related parameters\n"
  "    d  = 8.\n"
  "    p  = 1.2\n"
  "    dM = -22 (ms)\n"
  "    dV = 5 (ms)\n"
  "    ptau = 10 (ms)\n"
  "    pi = 3.14159\n"
  "    wmax = 0.005\n"
  "    wmin = 0.00001\n"
  "    pon  = 1\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    v (mV)\n"
  "    dt (ms)\n"
  "    i (nA)\n"
  "    tpost (ms)\n"
  "    wfampa\n"
  "    factorampa\n"
  "    wfnmda\n"
  "    factornmda\n"
  "    \n"
  "}\n"
  "\n"
  "STATE {\n"
  "    Aampa (uS)\n"
  "    Bampa (uS)\n"
  "    Anmda (uS)\n"
  "    Bnmda (uS)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    LOCAL tpampa, tpnmda\n"
  "    if (tau1ampa/tau2ampa > .9999) {\n"
  "        tau1ampa = .9999*tau2ampa\n"
  "    }\n"
  "    Aampa = 0\n"
  "    Bampa = 0\n"
  "    tpampa = (tau1ampa*tau2ampa)/(tau2ampa - tau1ampa) * log(tau2ampa/tau1ampa)\n"
  "    factorampa = -exp(-tpampa/tau1ampa) + exp(-tpampa/tau2ampa)\n"
  "    factorampa = 1/factorampa\n"
  "    \n"
  "    if (tau1nmda/tau2nmda > .9999) {\n"
  "        tau1nmda = 0.9999*tau2nmda\n"
  "    }\n"
  "    Anmda = 0\n"
  "    Bnmda = 0\n"
  "    tpnmda = (tau1nmda*tau2nmda)/(tau2nmda - tau1nmda) * log(tau2nmda/tau1nmda)\n"
  "    factornmda = -exp(-tpnmda/tau1nmda) + exp(-tpnmda/tau2nmda)\n"
  "    factornmda = 1/factornmda\n"
  "    Mgblock(v)\n"
  "    \n"
  "    tpost = -1e9\n"
  "    net_send(0, 1)\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE state METHOD cnexp\n"
  "    i = (Bampa - Aampa)*(v-e) + (Bnmda - Anmda)*Mgblock(v)*(v-e)\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "    Aampa' = -Aampa/tau1ampa\n"
  "    Bampa' = -Bampa/tau2ampa\n"
  "    \n"
  "    Anmda' = -Anmda/tau1nmda\n"
  "    Bnmda' = -Bnmda/tau2nmda\n"
  "}\n"
  "\n"
  "NET_RECEIVE(weightampa, weightnmda, A, tpre (ms)) {\n"
  "    INITIAL { \n"
  "        weightnmda = weightnmda\n"
  "        A = 0\n"
  "        tpre = -1e9\n"
  "    }\n"
  "    if (flag == 0) { : presynpatic spike detection\n"
  "        if (pon == 1) {\n"
  "            :printf(\"gid: %g. presynaptic spike time: %g. ampa weight: %g\\. nmda weight: %g. \\n\", gid, t, (weightampa+A), weightnmda)\n"
  "            wfampa = (weightampa+A)*factorampa       \n"
  "            tpre = t\n"
  "            A = A * (1-(d*exp(-((tpost-t)-dM)^2/(2*dV*dV))) /(sqrt(2*pi)*dV))\n"
  "        } \n"
  "        else {\n"
  "            wfampa = weightampa*factorampa\n"
  "        }\n"
  "        Aampa = Aampa + wfampa\n"
  "        Bampa = Bampa + wfampa\n"
  "\n"
  "        wfnmda = weightnmda*factornmda\n"
  "        Anmda = Anmda + wfnmda\n"
  "        Bnmda = Bnmda + wfnmda\n"
  "\n"
  "    } \n"
  "    else if (flag == 1) { : from INITIAL block, will detect postsynaptic spike if voltage > thresh and send flag = 2\n"
  "        WATCH (v > thresh) 2\n"
  "    } \n"
  "    \n"
  "    else if (flag == 2) { : post synaptic spike detection\n"
  "        if (pon == 1) {\n"
  "            :printf(\"gid: %g postsynaptic spike detected\\n\", gid)\n"
  "            tpost = t\n"
  "            FOR_NETCONS(w1, w2, A1, tpr) { : also can hide NET_RECEIVE args\n"
  "                :printf(\"netcon loop gid: %g. tpre-t: %g\\n\", gid, (tpr-t))\n"
  "                :gitprintf(\"post-pre: %g. andd %g\\n\",tpost-tpr, wmax-w1-A1)\n"
  "                A1 = A1 + (wmax-w1-A1)*p*exp((tpr - t)/ptau)\n"
  "            }\n"
  "        }\n"
  "    }\n"
  "}\n"
  "\n"
  "FUNCTION Mgblock(v(mV)) {\n"
  "    : from Wang et. al 2002\n"
  "    Mgblock = 1 / (1 + eta * Mg * exp( - (gamma * v)))\n"
  "}\n"
  ;
#endif
