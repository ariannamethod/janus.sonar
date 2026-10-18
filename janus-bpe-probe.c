/*
 * janus-bpe-probe.c — backward probes for janus-bpe.c
 *
 *   cc janus-bpe-probe.c -O2 -lm -o probe                                  (this tree's janus-bpe.c)
 *   cc janus-bpe-probe.c -O2 -lm -DJB_SRC='"/abs/path/old/janus-bpe.c"' -o probe_old
 *   ./probe fwd                        forward fingerprint on fixed inputs (must match across revisions)
 *   ./probe grad [T] [eps] [n]         gradcheck: f64-reference central differences, n params per class over all blocks
 *   ./probe step data.txt              one real train_bpe() step, max|dw| per class
 *   ./probe train data.txt N [lr]      fixed seed, fixed batch stream, 90/10 split: step|train|val
 */
#ifndef JB_SRC
#define JB_SRC "janus-bpe.c"
#endif
#define main janus_bpe_main
#include JB_SRC
#undef main

#define NCLS 17
static const char*CN[NCLS]={"tok_emb","pos_emb","rms1","wq","wk","wv","wr","wvr","wj","gate",
                            "wo","rms2","w_gate","w_up","w_down","rms_f","out_w"};
static int cls_blk(int c){return c>=2&&c<=14;}
static float*cls_ptr(Ptrs*p,int c,int b){
    switch(c){case 0:return p->tok_emb;case 1:return p->pos_emb;case 2:return p->rms1[b];
    case 3:return p->wq[b];case 4:return p->wk[b];case 5:return p->wv[b];case 6:return p->wr[b];
    case 7:return p->wvr[b];case 8:return p->wj[b];case 9:return p->gate[b];case 10:return p->wo[b];
    case 11:return p->rms2[b];case 12:return p->w_gate[b];case 13:return p->w_up[b];case 14:return p->w_down[b];
    case 15:return p->rms_f;default:return p->out_w;}
}
static int cls_n(int c){
    switch(c){case 0:return BPE_VOCAB*DIM;case 1:return MAX_T*DIM;case 2:case 11:case 15:return DIM;
    case 3:case 4:case 5:case 7:return HEADS*DIM*HEAD_DIM;case 6:return HEADS*DIM*MAX_T;
    case 8:case 10:return DIM*DIM;case 9:return HEADS*3;case 12:case 13:return DIM*MLP_DIM;
    case 14:return MLP_DIM*DIM;default:return DIM*BPE_VOCAB;}
}

/* CE of the forward's logits, reduced in double (forward with tgt=NULL leaves raw logits in a->lg) */
static double loss_d(Model*m,Acts*a,int*tok,int*tgt,int T){
    forward(&m->w,a,tok,NULL,T);double L=0;
    for(int t=0;t<T;t++){const float*l=a->lg+(size_t)t*BPE_VOCAB;double mx=l[0],s=0;
    for(int v=1;v<BPE_VOCAB;v++)if(l[v]>mx)mx=l[v];
    for(int v=0;v<BPE_VOCAB;v++)s+=exp((double)l[v]-mx);L+=mx+log(s)-l[tgt[t]];}
    return L/T;
}

/* Double transcription of forward(): same masks, same wr stride MAX_T, same f32 constants (sc, cd, dt, 1e-5f, 1e-6f).
   f32 central differences sit on a ~1e-7 rounding floor in L (|g|~1e-4..1e-5 drowns); this one does not.
   probe_grad refuses to run if it disagrees with forward() beyond f32 precision. */
static void rmm_w(double*C,const double*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[p*n+j];C[i*n+j]=s;}
}
static void rmm_w_ld(double*C,const double*A,const float*B,int m,int k,int n,int ld){
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[p*ld+j];C[i*n+j]=s;}
}
static void rmm(double*C,const double*A,const double*B,int m,int k,int n){
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[p*n+j];C[i*n+j]=s;}
}
static void rrms(double*o,const double*x,const float*g,int T){
    for(int t=0;t<T;t++){double ss=0;for(int e=0;e<DIM;e++)ss+=x[t*DIM+e]*x[t*DIM+e];
    double inv=1/sqrt(ss/DIM+1e-5f);for(int e=0;e<DIM;e++)o[t*DIM+e]=g[e]*x[t*DIM+e]*inv;}
}
static void rcsm(double*a,int T){
    for(int i=0;i<T;i++){double mx=a[i*T],s=0;for(int j=1;j<=i;j++)if(a[i*T+j]>mx)mx=a[i*T+j];
    for(int j=0;j<T;j++){a[i*T+j]=j<=i?exp(a[i*T+j]-mx):0;s+=a[i*T+j];}for(int j=0;j<=i;j++)a[i*T+j]/=s;}
}
static double ref_loss(Ptrs*w,const int*tok,const int*tgt,int T){
    static double x[MAX_T*DIM],rm[MAX_T*DIM],cat[MAX_T*DIM],y[MAX_T*DIM],je[MAX_T*DIM],
        q[MAX_T*HEAD_DIM],k[MAX_T*HEAD_DIM],v[MAX_T*HEAD_DIM],o[MAX_T*HEAD_DIM],
        at[MAX_T*MAX_T],ja[MAX_T*MAX_T],mg[MAX_T*MLP_DIM],mu[MAX_T*MLP_DIM],js[MAX_T],lg[BPE_VOCAB];
    const double hs=1/sqrtf((float)HEAD_DIM),cd=1+0.5f*calendar_dissonance(calendar_days_since_epoch()),dt=1+AML.prophecy_debt;
    for(int t=0;t<T;t++)for(int e=0;e<DIM;e++)x[t*DIM+e]=(double)w->tok_emb[tok[t]*DIM+e]+w->pos_emb[t*DIM+e];
    for(int b=0;b<BLOCKS;b++){
        rrms(rm,x,w->rms1[b],T);memset(cat,0,sizeof cat);
        for(int h=0;h<HEADS;h++){
            double gl[3],gm=w->gate[b][h*3],gs=0;for(int i=1;i<3;i++)if(w->gate[b][h*3+i]>gm)gm=w->gate[b][h*3+i];
            for(int i=0;i<3;i++){gl[i]=exp(w->gate[b][h*3+i]-gm);gs+=gl[i];}for(int i=0;i<3;i++)gl[i]/=gs;
            rmm_w(q,rm,w->wq[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);rmm_w(k,rm,w->wk[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            rmm_w(v,rm,w->wv[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            for(int i=0;i<T;i++)for(int j=0;j<=i;j++){double s=0;for(int d=0;d<HEAD_DIM;d++)s+=q[i*HEAD_DIM+d]*k[j*HEAD_DIM+d];at[i*T+j]=s*hs;}
            rcsm(at,T);rmm(o,at,v,T,T,HEAD_DIM);
            for(int t=0;t<T;t++)for(int d=0;d<HEAD_DIM;d++)cat[t*DIM+h*HEAD_DIM+d]+=gl[0]*o[t*HEAD_DIM+d];
            rmm_w(v,rm,w->wvr[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            rmm_w_ld(at,rm,w->wr[b]+h*DIM*MAX_T,T,DIM,T,MAX_T);for(int i=0;i<T*T;i++)at[i]*=hs;
            rcsm(at,T);rmm(o,at,v,T,T,HEAD_DIM);
            for(int t=0;t<T;t++)for(int d=0;d<HEAD_DIM;d++)cat[t*DIM+h*HEAD_DIM+d]+=gl[1]*o[t*HEAD_DIM+d];
            if(h==0){const float*wj=w->wj[b];
                for(int t=0;t<T;t++){const double*xt=rm+t*DIM;double*pr=je+t*DIM,nm=0,sv=0;
                for(int i=0;i<DIM;i++){double s=0;for(int j=0;j<DIM;j++)s+=wj[i*DIM+j]*xt[j];pr[i]=s;nm+=s*s;}
                for(int i=0;i<DIM;i++){double s=0;for(int j=0;j<DIM;j++)s+=wj[j*DIM+i]*pr[j];sv+=xt[i]*s;}
                js[t]=sv/(sqrt(nm)+1e-6f)*cd;}
                for(int i=0;i<T;i++)for(int j=0;j<=i;j++)ja[i*T+j]=js[i]*js[j]/dt;rcsm(ja,T);}
            for(int t=0;t<T;t++)for(int d=0;d<HEAD_DIM;d++){double s=0;
                for(int j=0;j<=t;j++)s+=ja[t*T+j]*je[j*DIM+h*HEAD_DIM+d];cat[t*DIM+h*HEAD_DIM+d]+=gl[2]*s;}
        }
        rmm_w(y,cat,w->wo[b],T,DIM,DIM);for(int i=0;i<T*DIM;i++)x[i]+=y[i];
        rrms(rm,x,w->rms2[b],T);rmm_w(mg,rm,w->w_gate[b],T,DIM,MLP_DIM);rmm_w(mu,rm,w->w_up[b],T,DIM,MLP_DIM);
        for(int i=0;i<T*MLP_DIM;i++)mg[i]=(mg[i]>-20?mg[i]/(1+exp(-mg[i])):0)*mu[i];
        rmm_w(y,mg,w->w_down[b],T,MLP_DIM,DIM);for(int i=0;i<T*DIM;i++)x[i]+=y[i];
    }
    rrms(rm,x,w->rms_f,T);double L=0;
    for(int t=0;t<T;t++){rmm_w(lg,rm+t*DIM,w->out_w,1,DIM,BPE_VOCAB);double mx=lg[0],s=0;
    for(int v2=1;v2<BPE_VOCAB;v2++)if(lg[v2]>mx)mx=lg[v2];
    for(int v2=0;v2<BPE_VOCAB;v2++)s+=exp(lg[v2]-mx);L+=mx+log(s)-lg[tgt[t]];}
    return L/T;
}

static int probe_fwd(void){
    srand(1);Model m;model_init(&m);Acts a;acts_alloc(&a);
    int Ts[2]={32,MAX_T};
    for(int r=0;r<2;r++){int T=Ts[r],tok[MAX_T],tgt[MAX_T];srand(7);
    for(int t=0;t<T;t++){tok[t]=rand()%BPE_VOCAB;tgt[t]=rand()%BPE_VOCAB;}
    float L=forward(&m.w,&a,tok,tgt,T);uint64_t hsh=1469598103934665603ULL;
    const unsigned char*p=(const unsigned char*)a.lg;
    for(size_t i=0;i<(size_t)T*BPE_VOCAB*4;i++){hsh^=p[i];hsh*=1099511628211ULL;}
    printf("fwd T=%d loss=%.9g probs_fnv64=%016llx\n",T,L,(unsigned long long)hsh);}
    acts_free(&a);model_free(&m);return 0;
}

/* Gradcheck point: generic weights — at init scale 0.0025 every softmax is ~uniform and gates sit at 1/3,
   so their backward terms are barely exercised. wj at 0.01 keeps janus logits sc_i*sc_j ~ O(1), not saturated. */
static void gc_point(Model*m){
    for(int i=0;i<m->n_params;i++)m->data[i]=randn()*0.1f;
    for(int b=0;b<BLOCKS;b++){for(int i=0;i<DIM*DIM;i++)m->w.wj[b][i]*=0.1f;
    for(int i=0;i<HEADS*3;i++)m->w.gate[b][i]=randn()*0.5f;
    for(int e=0;e<DIM;e++){m->w.rms1[b][e]=1+0.1f*randn();m->w.rms2[b][e]=1+0.1f*randn();}}
    for(int e=0;e<DIM;e++)m->w.rms_f[e]=1+0.1f*randn();
}

static double relerr(double a,double b){double d=fmax(fabs(a),fabs(b));return d>0?fabs(a-b)/d:0;}

static int probe_grad(int T,float eps,int n){
    if(T<2||T>MAX_T)T=MAX_T;
    srand(1);Model m;model_init(&m);Acts a;acts_alloc(&a);
    int tok[MAX_T],tgt[MAX_T];for(int t=0;t<T;t++){tok[t]=rand()%BPE_VOCAB;tgt[t]=rand()%BPE_VOCAB;}
    double R0=ref_loss(&m.w,tok,tgt,T),F0=loss_d(&m,&a,tok,tgt,T);gc_point(&m);
    memset(m.grad,0,m.n_params*sizeof(float));
    float L=forward(&m.w,&a,tok,tgt,T);backward(&m.w,&m.g,&a,tok,tgt,T);
    double R=ref_loss(&m.w,tok,tgt,T),F=loss_d(&m,&a,tok,tgt,T);
    printf("gradcheck T=%d n=%d/class  gate: f64-ref central diff eps=%g | context: f32 forward() central diff eps=1e-2\n",T,n,eps);
    printf("ref check  init point: forward=%.9f ref=%.9f rel=%.2e | gc point: forward=%.9f ref=%.9f rel=%.2e (forward ret %.6f)\n",
           F0,R0,relerr(F0,R0),F,R,relerr(F,R),L);
    if(relerr(F0,R0)>1e-5||relerr(F,R)>1e-5){printf("REF MISMATCH: f64 transcription disagrees with forward()\n");return 3;}
    printf("%-7s %3s %7s %14s %14s %10s %14s %10s\n","class","blk","idx","analytic","num_f64","rel","num_f32","rel_f32");
    int fails=0;
    struct{double mx,nmin,nmax;int zero;}S[NCLS];
    for(int c=0;c<NCLS;c++){S[c].mx=0;S[c].nmin=1e30;S[c].nmax=0;S[c].zero=0;
    for(int i=0;i<n;i++){int b=cls_blk(c)?i%BLOCKS:0,idx;
        if(c==0)idx=tok[rand()%T]*DIM+rand()%DIM;
        else if(c==1)idx=(rand()%T)*DIM+rand()%DIM;
        else if(c==6)idx=(rand()%HEADS)*DIM*MAX_T+(rand()%DIM)*MAX_T+rand()%T;   /* forward reads wr_h[p*MAX_T+j], j<T */
        else if(c==16)idx=(rand()%DIM)*BPE_VOCAB+tgt[rand()%T];
        else idx=rand()%cls_n(c);
        float*w=cls_ptr(&m.w,c,b)+idx;double an=cls_ptr(&m.g,c,b)[idx];float w0=*w;
        *w=w0+eps;float wp=*w;double Rp=ref_loss(&m.w,tok,tgt,T);
        *w=w0-eps;float wm=*w;double Rm=ref_loss(&m.w,tok,tgt,T);
        *w=w0+1e-2f;float fp=*w;double Fp=loss_d(&m,&a,tok,tgt,T);
        *w=w0-1e-2f;float fm=*w;double Fm=loss_d(&m,&a,tok,tgt,T);*w=w0;
        double nu=(Rp-Rm)/((double)wp-(double)wm),nf=(Fp-Fm)/((double)fp-(double)fm),rel=relerr(an,nu);
        if(an==0&&fabs(nu)>0)S[c].zero++;
        if(rel>S[c].mx)S[c].mx=rel;
        if(fabs(nu)<S[c].nmin)S[c].nmin=fabs(nu);
        if(fabs(nu)>S[c].nmax)S[c].nmax=fabs(nu);
        printf("%-7s %3d %7d %14.6e %14.6e %10.3e %14.6e %10.3e%s\n",CN[c],b,idx,an,nu,rel,nf,relerr(an,nf),rel<1e-2?"":"  <-- FAIL");}
    }
    printf("\n%-7s %10s %12s %12s %8s %s\n","class","max_rel","min|num|","max|num|","an==0","verdict");
    for(int c=0;c<NCLS;c++){int ok=S[c].mx<1e-2;fails+=!ok;
    printf("%-7s %10.3e %12.3e %12.3e %5d/%-2d %s\n",CN[c],S[c].mx,S[c].nmin,S[c].nmax,S[c].zero,n,ok?"PASS":"FAIL");}
    printf("%s: %d/%d classes fail (criterion max rel < 1e-2)\n",fails?"GRADCHECK FAIL":"GRADCHECK PASS",fails,NCLS);
    acts_free(&a);model_free(&m);return fails?1:0;
}

static int probe_step(const char*path){
    srand(1);Model m;model_init(&m);
    float*w0=malloc(m.n_params*sizeof(float));memcpy(w0,m.data,m.n_params*sizeof(float));
    train_bpe(&m,path,1,3e-4f);
    Ptrs W0;assign_ptrs(&W0,w0);int dead=0;
    printf("\n%-7s %12s %10s %10s\n","class","max|dw|","moved","total");
    for(int c=0;c<NCLS;c++){double mx=0;long mv=0,tot=0;
    for(int b=0;b<(cls_blk(c)?BLOCKS:1);b++){float*p=cls_ptr(&m.w,c,b),*q=cls_ptr(&W0,c,b);
    for(int i=0;i<cls_n(c);i++){double d=fabs((double)p[i]-q[i]);if(d>mx)mx=d;mv+=d>0;tot++;}}
    dead+=mx==0;printf("%-7s %12.4e %10ld %10ld%s\n",CN[c],mx,mv,tot,mx==0?"  <-- DEAD":"");}
    printf("%d/%d classes did not move\n",dead,NCLS);
    free(w0);model_free(&m);return 0;
}

static uint32_t xs=0x9E3779B9u;
static uint32_t xs32(void){xs^=xs<<13;xs^=xs>>17;xs^=xs<<5;return xs;}

static int probe_train(const char*path,int steps,float lr){
    FILE*f=fopen(path,"r");if(!f){fprintf(stderr,"cannot open %s\n",path);return 1;}
    fseek(f,0,SEEK_END);long fsz=ftell(f);fseek(f,0,SEEK_SET);
    unsigned char*raw=malloc(fsz);if(fread(raw,1,fsz,f)!=(size_t)fsz){fclose(f);return 1;}fclose(f);
    bpe_learn_merges(raw,fsz,BPE_VOCAB-256);
    int*bd=malloc(fsz*sizeof(int));int bl=bpe_encode(raw,fsz,bd,fsz);free(raw);
    const int T=MAX_T,NV=16;int ntr=bl*9/10,tok[MAX_T],tgt[MAX_T];
    printf("[probe] %d tokens, train [0,%d) val [%d,%d), %d val windows, GRAD_ACCUM=%d lr=%g\n",bl,ntr,ntr,bl,NV,GRAD_ACCUM,lr);
    srand(1);Model m;model_init(&m);Acts a;acts_alloc(&a);
    double acc=0;int nacc=0;clock_t t0=clock();
    for(int step=1;step<=steps;step++){
        memset(m.grad,0,m.n_params*sizeof(float));float sl=0;
        for(int ga=0;ga<GRAD_ACCUM;ga++){int off=xs32()%(uint32_t)(ntr-T-1);
            for(int t=0;t<T;t++){tok[t]=bd[off+t]%BPE_VOCAB;tgt[t]=bd[off+t+1]%BPE_VOCAB;}
            sl+=forward(&m.w,&a,tok,tgt,T);backward(&m.w,&m.g,&a,tok,tgt,T);}
        sl/=GRAD_ACCUM;for(int i=0;i<m.n_params;i++)m.grad[i]*=1.0f/GRAD_ACCUM;
        chuck_observe(sl);chuck_update(m.data,m.grad,m.cm,m.cv,m.n_params,lr);
        acc+=sl;nacc++;
        if(step==1||step%25==0){double vl=0;
            for(int k=0;k<NV;k++){int off=ntr+k*((bl-ntr-T-1)/NV);
            for(int t=0;t<T;t++){tok[t]=bd[off+t]%BPE_VOCAB;tgt[t]=bd[off+t+1]%BPE_VOCAB;}
            vl+=forward(&m.w,&a,tok,tgt,T);}
            printf("step %4d | train %.4f | val %.4f | %.1fs\n",step,acc/nacc,vl/NV,(double)(clock()-t0)/CLOCKS_PER_SEC);
            fflush(stdout);acc=0;nacc=0;}
    }
    acts_free(&a);model_free(&m);free(bd);return 0;
}

int main(int argc,char**argv){
    calendar_init();metajanus_init();
    if(argc>=2&&!strcmp(argv[1],"fwd"))return probe_fwd();
    if(argc>=2&&!strcmp(argv[1],"grad"))
        return probe_grad(argc>2?atoi(argv[2]):32,argc>3?(float)atof(argv[3]):1e-4f,argc>4?atoi(argv[4]):12);
    if(argc>=3&&!strcmp(argv[1],"step"))return probe_step(argv[2]);
    if(argc>=4&&!strcmp(argv[1],"train"))return probe_train(argv[2],atoi(argv[3]),argc>4?(float)atof(argv[4]):3e-4f);
    fprintf(stderr,"usage: %s fwd | grad [T] [eps] [n] | step data.txt | train data.txt N [lr]\n",argv[0]);return 2;
}
