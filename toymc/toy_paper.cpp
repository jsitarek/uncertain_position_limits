// toy MC comparison of 3 methods:
// 1) least-constraining Rolke U.L.
// 2) approach like what IC is using in https://arxiv.org/pdf/1908.07706
// 3) bayesian

#include <TH1.h>
#include <TH2.h>
#include <TF1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <TLine.h>
#include <TLegend.h>
#include <TRolke.h>
#include <TPad.h>
#include <TFile.h>

#include <iostream>
#include <fstream>
#include "MMath.h"

#include "Minuit2/Minuit2Minimizer.h"
#include "Math/GSLMinimizer.h"
#include "Math/GSLNLSMinimizer.h"
#include "Math/Functor.h"

using namespace std;

// globals
// TF1 *fbgd; // background
// TF1 *faeff; // effective area
TH1 *hgw; // description of ToO PDF
TH1 *hbgd; // background
TH1 *haeff; // effective area

TH1F *hon=0;
TH1F *hoff=0;

const Double_t xmin=-10, xmax=10;
const Int_t nbins=40;
const Double_t offmulti=5; //number of off positions (for background estimation)
const Double_t conf=0.95; // confidence level of U.L.
const Bool_t useLiMalike=kTRUE; // if Li&Ma should be used for TS instead of numerical fits 
const Bool_t usefreqflux=kFALSE; // return the flux of the maximum TS instead of TS in frequentist method
TH1D *ghprob=0; // global to the probability distribution

// === preparations ===
void prep_irf()
{
  cout<<"=== preparing setup ==="<<endl;
  TF1 *fgw = new TF1 ("fgw", "gaus", xmin, xmax);
  fgw->SetParameters(200, 0, 3.5);

  TCanvas *canprep = new TCanvas ("canprep", "", 640, 480);
  canprep->SetTopMargin(0.01);
  canprep->SetRightMargin(0.01);
  TF1 *faeff = new TF1 ("faeff", "100*(exp(-pow(x-6.,2)/(2*pow(2.,2)))+exp(-pow(x,2)/(2*pow(2.,2)))+exp(-pow(x+6.,2)/(2*pow(2.,2))))", xmin, xmax);
  TF1 *fbgd = new TF1 ("fbgd", "100*(exp(-pow(x-6.,2)/(2*pow(2.2,2)))+exp(-pow(x,2)/(2*pow(2.2,2)))+exp(-pow(x+6.,2)/(2*pow(2.2,2))))", xmin, xmax);
  // TF1 *faeff = new TF1("faeff", "100", xmin, xmax);
  // TF1 *fbgd = new TF1("fbgd", "100", xmin, xmax);
  hgw = new TH1F ("hgw", ";position;scaled GW probability", nbins, xmin, xmax);
  haeff = new TH1F ("haeff", ";position;Effective Area", nbins, xmin, xmax);
  hbgd = new TH1F ("hbgd", ";position;Background", nbins, xmin, xmax);
  hgw->SetLineWidth(2);
  haeff->SetLineWidth(2);
  hbgd->SetLineWidth(2);
  for (int i=1; i<=nbins; i++)
    {
      hgw->SetBinContent(i, fgw->Eval(hgw->GetBinCenter(i)));
      haeff->SetBinContent(i, faeff->Eval(haeff->GetBinCenter(i)));
      hbgd->SetBinContent(i, fbgd->Eval(hbgd->GetBinCenter(i)));
    }
  hgw->Scale(1000./hgw->Integral(1, nbins+1, "width"));
  hgw->SetStats(0);
  canprep->cd();
  hgw->SetLineColor(kBlack);
  haeff->SetLineColor(kGreen);
  haeff->SetLineStyle(kDotted);
  hbgd->SetLineColor(kBlue);
  hbgd->SetLineStyle(kDashed);
  hgw->Draw("hist C");
  hgw->GetXaxis()->SetTitle("Position");
  hgw->GetYaxis()->SetTitle("Aeff or Bgd or PDF (a.u.)");
  haeff->Draw("hist C same");
  hbgd->Draw("hist C same");

  TLegend *leg = new TLegend(0.35, 0.13, 0.74, 0.42);
  leg->AddEntry(hgw, "alert PDF [x1000]", "L");
  leg->AddEntry(hbgd, "Background acceptance", "L");
  leg->AddEntry(haeff, "Effective area", "L");
  leg->Draw();
}

// if hofforg is provided background is taken from previous simulations rather than from true values
void fillhists(Double_t flux, Double_t &truex, TH1F *hofforg=0, Bool_t quiet=kFALSE)
{
  if (hon) delete hon;
  if (hoff) delete hoff;
  hon = new TH1F ("hon", ";position;number of events ", nbins, xmin, xmax);
  hoff = (TH1F*)hon->Clone("hoff");
  hon->SetStats(0);
  hoff->SetStats(0);
  hoff->Sumw2();
  hon->SetLineColor(kRed);
  hoff->SetLineColor(kBlue);
  hoff->SetLineStyle(kDashed);
  hoff->SetLineWidth(3);

  for (int i=1; i<=nbins; i++)
    {
      Double_t x = hon->GetBinCenter(i);
      Double_t off = hbgd->GetBinContent(hbgd->FindBin(x));
      if (hofforg)
	off = hofforg->GetBinContent(hofforg->FindBin(x));
      Double_t on = off; 
      on = gRandom->Poisson(off);
      off = gRandom->Poisson(off*offmulti)/offmulti;
   
      hoff->SetBinContent(i,off);
      hoff->SetBinError(i, sqrt(off)/sqrt(offmulti));
      hon->SetBinContent(i,on);
    }
  if (flux<=0)
    return;
  truex = hgw->GetRandom();
  Double_t ntrue=haeff->GetBinContent(haeff->FindBin(truex))*flux;
  Int_t ix = hon->FindBin(truex);
  if (!quiet)
    cout<<"source at x="<<truex<<", ntrue = "<<ntrue<<endl;
  hon->Fill(truex, gRandom->Poisson(ntrue));
  hon->SetBinError(ix, sqrt(hon->GetBinContent(ix)));
}


// calculates the best B estimation assuming that the source has a strength of S
Double_t calcBest(Double_t ON, Double_t OFF, Double_t NPOS, Double_t S)
{
  Double_t delta =pow(S*(1+NPOS) - ON - OFF, 2) + 4 *(NPOS+1) *S * OFF;
  Double_t B= (-(S*(1+NPOS) - ON - OFF) + sqrt(delta)) / (2*(NPOS+1));
  // Double_t x2= (-(S*(1+NPOS) - ON - OFF) - sqrt(delta)) / (2*(NPOS+1));
  return B;
}


Double_t logpoisson(Double_t x, Double_t par)
{
  if (par < 0)
    return TMath::QuietNaN();
  if (x < 0)
    return -1.e10;
  else if (x == 0.0 )
    return -par;
  else
    return x * log(par) - TMath::LnGamma(x + 1.) - par;
}

// === Bayesian ===
// probability of getting the data with assuming position of x0 and a flux of f0
Double_t p_d_true(Double_t x0, Double_t f0, Bool_t singlebin=kFALSE)
{
  Int_t ix = hon->FindBin(x0);
  Double_t n0= haeff->GetBinContent(haeff->FindBin(x0))*f0;
  Double_t pall=1;
  for (int i=1; i<=nbins; i++)
    {
      Double_t x = hon->GetBinCenter(i);
      Double_t on=hon->GetBinContent(i);
      Double_t don=hon->GetBinError(i);
      Double_t off=hoff->GetBinContent(i);
      Double_t model=off;
      Double_t dmodel=hoff->GetBinError(i);
      if (i == ix)
	{
	  model+=n0;
	  dmodel = sqrt(dmodel*dmodel + n0);
	}
      else if (singlebin)
	continue;
      Double_t S = model - off;
      Double_t B = calcBest(on, off*offmulti, offmulti, S);
      Double_t probOFF = TMath::Poisson(off*offmulti, B*offmulti);
      Double_t probON = TMath::Poisson(on, S+B);
      Double_t prob = probOFF * probON;
      pall*=prob;
    }
  pall*=hgw->GetBinContent(hgw->FindBin(x0));
  return pall;
}

TH1D *hhh=0;
Double_t bayes_scan(Double_t f0min, Double_t f0max, Double_t &bayesmax, Double_t &bayeslo, Double_t &bayeshi, Double_t &giacomoul, Bool_t drawit=kFALSE, Bool_t singlebin=kFALSE)
{
  Int_t nf0=200; //number of bins to scan the flux 
  // calculate the last bin
  TH2D *hprob = new TH2D ("hprob", "Probability;Position;Flux", nbins, xmin, xmax, nf0, f0min, f0max);
  for (int i1=1; i1<=nbins; i1++)
    for (int i2=1; i2<=nf0; i2++)
      {
	Double_t x0=hprob->GetXaxis()->GetBinCenter(i1);
	Double_t f0=hprob->GetYaxis()->GetBinCenter(i2);
	hprob->SetBinContent(i1, i2, p_d_true(x0, f0, singlebin));
      }
  cout<<"Probability normalization: "<<hprob->GetSum()<<endl;
  // normalize probability
  hprob->Scale(1./hprob->GetSum());
  TH1D *hprobx0 = hprob->ProjectionX();
  TH1D *hprobf0 = hprob->ProjectionY();
  hprobf0->GetYaxis()->SetTitle("Probability");
  hhh=hprobf0;
  bayesmax= hprobf0->GetBinCenter(hprobf0->GetMaximumBin());
  
  Double_t intprob=0.5, bayesul;
  hprobf0->GetQuantiles(1, &bayesmax, &intprob); // take median
  intprob=(1-0.68)/2.;
  hprobf0->GetQuantiles(1, &bayeslo, &intprob); // take low edge
  intprob=1-intprob;
  hprobf0->GetQuantiles(1, &bayeshi, &intprob); // take high edge
  hprobf0->GetQuantiles(1, &bayesul, &conf);

  ghprob=hprobf0; // setting global
  // Giacomos's idea
  TGraph *ggiacomo = new TGraph ();
  for (int i=1; i<=hprobf0->GetNbinsX(); i++)
    {
      Double_t p=hprobf0->GetBinContent(i);
      Double_t f=hprobf0->GetBinCenter(i);
      ggiacomo->SetPoint(i-1, f, -2*log(p));
    }
  TF1 ffit("ffit", "pol2", bayesmax-(bayesul-bayesmax), bayesul);
  ggiacomo->Fit(&ffit, drawit?"R":"RQ");
  Double_t *pars = ggiacomo->GetFunction("ffit")->GetParameters();
  Double_t min= -pars[1]/(2*pars[2]);
  Double_t ymin=ggiacomo->GetFunction("ffit")->Eval(min);
  Double_t a=pars[2], b=pars[1], c=pars[0]-ymin-2.71;
  giacomoul = (-b+sqrt(b*b-4*a*c))/(2*a);
  cout<<"ymin = "<<ymin<<" at "<<min<<", giacomo's ul="<<giacomoul<<", diff="<<ggiacomo->GetFunction("ffit")->Eval(giacomoul)-ymin<<endl;
  
  
  if (!drawit)
    {
      delete hprob;
      delete ggiacomo;
    }
  else
    {
      TCanvas *can2 = new TCanvas ("can2", "", 1024, 768);
      can2->Divide(2,2, 0.001, 0.001);
      can2->cd(1);
      hprob->Draw("colz");
      can2->GetPad(1)->SetLogz();
      can2->cd(2);
      hprobf0->Draw("hist");
      TLine *l1 = new TLine (bayesmax, hprobf0->GetMinimum(), bayesmax, hprobf0->GetMaximum());
      TLine *l2 = new TLine (bayesul, hprobf0->GetMinimum(), bayesul, hprobf0->GetMaximum());
      l1->SetLineStyle(kDotted);
      l1->Draw();
      l2->Draw();
      TLine *l3 = new TLine (giacomoul, hprobf0->GetMinimum(), giacomoul, hprobf0->GetMaximum());
      l3->SetLineColor(kRed);
      l3->SetLineStyle(kDashed);
      l3->Draw();
      TLegend *leg = new TLegend (0.56, 0.52, 0.99, 0.76);
      leg->AddEntry(l1, Form("Max prob at F=%.2f",bayesmax), "L");
      leg->AddEntry(l2, Form("%.0f %% C.L. U.L. at F=%.2f",conf*100., bayesul), "L");
      leg->AddEntry(l3, Form("-2ln(p/p_{max})=2.71 at F=%.2f", giacomoul), "L");
      leg->Draw();
      can2->GetPad(2)->SetLogy();
      can2->cd(3);
      hprobx0->Draw("hist");
      can2->GetPad(3)->SetLogy();
      can2->cd(4);
      can2->GetPad(4)->SetLogy();
      ggiacomo->Draw("APL");
      ggiacomo->GetFunction ("ffit")->Draw("same");
      ggiacomo->GetXaxis()->SetTitle("Flux");
      ggiacomo->GetYaxis()->SetTitle("-2 ln p");
      TCanvas *cc = new TCanvas ("cc", "", 640, 480);
      hprob->SetTitle("");
      hprob->SetMinimum(hprob->GetMaximum()*1.e-5);
      cc->SetTopMargin(0.01);
      cc->SetRightMargin(0.14);
      cc->SetLogz();
      hprob->SetStats(0);
      hprob->Draw("colz");
      
    }
  cout<<"Maximum at "<<bayesmax<<", 68% interval: "<<bayeslo<<" to "<<bayeshi<<", "<<conf<<"C.L. U.L. at "<<bayesul<<endl;
  return bayesul;
}

// === frequentist likelihood ===
Double_t loglike(Int_t on, Int_t off, Double_t alphaoff, Double_t mu, Double_t b)
// on - measured on
// off - sum of off from all measured off positions
// alphaoff - number of off position
// mu - strenght of signal (in counts)
// b - background estimation
{
  // gaussian assumption
  Double_t logl=0;
  // logl+=-0.5*TMath::Power(on-mu-b,2)/(mu+b) - 0.5*log(TMath::TwoPi()*(mu+b)); // on
  // logl+=-0.5*TMath::Power(off-b*alphaoff,2)/(b*alphaoff) - 0.5*log(TMath::TwoPi()*b*alphaoff);

  // poissonian
  logl+=log(TMath::Poisson(on, mu+b));
  logl+=log(TMath::Poisson(off, alphaoff*b));
  // logl+=logpoisson(on, mu+b);
  // logl+=logpoisson(off, alphaoff*b);
  return logl;
}

double fcn(const double *pars)
{
  Double_t x = pars[0]; // position of the source
  Int_t ibin=hon->FindBin(x);
  Double_t on = hon->GetBinContent(ibin);
  Double_t alphaoff = pars[3];
  Double_t off = hoff->GetBinContent(ibin)*alphaoff;

  Double_t flux = pars[1];
  Double_t mu = flux *haeff->GetBinContent(haeff->FindBin(x));
  Double_t b = pars[2];
  Double_t llike=loglike(on, off, alphaoff, mu, b);
  return -llike;  // minus because we minimize it
}

// minimize over the histogram bins using numerical likelihood fits or Li&Ma analytical formula
Double_t scan_over_bins(Int_t ibin1, Int_t ibin2, Bool_t quiet=kFALSE)
{
  if (useLiMalike)
    {
      Int_t bestbin=-1;
      Double_t bestmin=-1.e9;
      for (int ibin = ibin1; ibin<=ibin2; ibin++)
	{
	  Double_t on = hon->GetBinContent(ibin);
	  Double_t offall=offmulti * hoff->GetBinContent(ibin);
	  Double_t sigma = MMath::SignificanceLiMaSigned(on, offall, 1/offmulti);
	  if (sigma<0) sigma=0;
	  Double_t lambda = sigma*sigma+2*log(hgw->GetBinContent(ibin)/hgw->GetSumOfWeights());
	  // lambda=sigma;
	  if (lambda>bestmin)
	    {
	      bestmin=lambda;
	      bestbin=ibin;
	    }
	}
      if (!quiet)
	cout<<"!!!min = "<<bestmin<<", for bin"<<bestbin<<" (x="<<hon->GetBinCenter(bestbin)<<")"<<endl;
      if (usefreqflux)
	return (hon->GetBinContent(bestbin)-hoff->GetBinContent(bestbin))/(haeff->GetBinContent (bestbin));
      return bestmin;
    }
  
  ROOT::Minuit2::Minuit2Minimizer minuit ( ROOT::Minuit2::kMigrad );
  // ROOT::Minuit2::Minuit2Minimizer minuit ( ROOT::Minuit2::kSimplex );
  minuit.SetMaxFunctionCalls(100000);
  minuit.SetMaxIterations(10000);
  minuit.SetTolerance(0.003);

  minuit.SetPrintLevel(0);
  const Int_t npars=4;

  ROOT::Math::Functor f (&fcn, npars);
  minuit.SetFunction(f);
  TString varname[12]={"pos", "flux", "b", "alphaoff"};

  Int_t bestbin=-1;
  Double_t bestmin=-1.e9;
  Double_t bestflux=-1;
  for (int ibin = ibin1; ibin<=ibin2; ibin++)
    {
      Double_t on = hon->GetBinContent(ibin);
      Double_t offall=offmulti * hoff->GetBinContent(ibin);      
      Double_t x = hon->GetBinCenter(ibin);
      Double_t pars[npars]={0};
      Double_t step[npars]={0};
      pars[0]=x;
      pars[1]=0;
      pars[2]=hoff->GetBinContent(ibin);
      pars[3]=offmulti;
      step[1]=0.03;
      step[2]=sqrt(pars[2]);
      for (Int_t ipar = 0; ipar < npars; ipar++)
	minuit.SetVariable(ipar, varname[ipar].Data(), pars[ipar], step[ipar]);
      minuit.FixVariable(0);
      minuit.FixVariable(3);
      // minuit.SetVariableLowerLimit(1,0);// no negative fluxes 
      Bool_t converged_src = minuit.Minimize();
      Double_t min_src=minuit.MinValue();
      Double_t flux = minuit.X()[1];

      // without the source we can compute this analitically 
      if (converged_src)
	{
	  Double_t Best = calcBest(on, offall, offmulti, 0);		
	  Double_t min_nosrc = -loglike(on, offall, offmulti, 0, Best);
	  Double_t lambda = 2*(min_nosrc-min_src)+2*log(hgw->GetBinContent(ibin)/hgw->GetSumOfWeights());

	  Double_t sigma = MMath::SignificanceLiMaSigned(on, offall, 1/offmulti);
	  Double_t thissigma=TMath::Sign(sqrt(fabs(2*(min_nosrc-min_src))),on-offall/offmulti); 
	  cout<<"sqrt(-2lnL)="<<thissigma<<", LiMa sigma = "<<sigma<<endl;
	  if (lambda>bestmin)
	    {
	      bestmin=lambda;
	      bestbin=ibin;
	      bestflux=flux;
	    }
	}
    }
  if (!quiet)
    cout<<"!!!flux = "<<bestflux<<", min = "<<bestmin<<", for bin"<<bestbin<<" (x="<<hon->GetBinCenter(bestbin)<<")"<<endl;
  return bestmin;
}

Double_t calc_rolke(Double_t *limits, Bool_t quiet=kTRUE)
{
  Double_t conf2s=1-(1-conf)*2; // 1 sided ==> 2 sided
  cout<<"1 sided inverval of "<<conf<<" corresponds to 2sided of "<<conf2s<<endl;
  TRolke trolke (conf2s);
  Double_t worstlimit=-1;
  for (int i=1; i<=nbins; i++)
    {
      Double_t x = hon->GetBinCenter(i);
      Double_t aeff = haeff->GetBinContent(haeff->FindBin(x));
      Double_t on=hon->GetBinContent(i);
      Double_t off=hoff->GetBinContent(i);
      Double_t doff=hoff->GetBinError(i);
      trolke.SetGaussBkgKnownEff(on, off, doff, 1.);
      limits[i-1]=trolke.GetUpperLimit()/aeff;
      if (limits[i-1]>worstlimit)
	worstlimit=limits[i-1];
    }

  return worstlimit;
}

Double_t likelimits(Double_t fluxmin, TH1F *hofforg, Bool_t quiet=kTRUE)
{
  Int_t ibin1=1, ibin2=nbins; // full range
  Double_t min0 = scan_over_bins(ibin1, ibin2, kFALSE); // data TS
  Int_t n0=10000;// number of noise-only
  TH1F *hlambda = new TH1F ("hlambda", "Distribution for no signal;lambda;number of realizations", 100,
			    usefreqflux?-1:-10, usefreqflux?2:20);
  
  // now make a noise scan
  Int_t nchance=0;
  Double_t dummyx=0;
  for (int i=0; i<n0; i++)
    {
      fillhists(0, dummyx, hofforg);
      Double_t min = scan_over_bins(ibin1, ibin2, kTRUE);
      hlambda->Fill(min);
      if (min>min0)
   	nchance++;
     }
  Double_t q=0.5, median;
  hlambda->ComputeIntegral();
  hlambda->GetQuantiles(1, &median, &q);
  cout<<"chance probability for min="<<min0<<" = "<<100.*nchance/n0<<" %"<<endl;
  if (min0<median)
    {
      cout<<"too negative fluctuation using median="<<median<<" instead of "<<min0<<endl;
      min0=median;
    }
  if (!quiet)
    {
      TCanvas *c4 = new TCanvas ("c4", "", 640, 480);
      hlambda->Draw();
    }
  TGraphErrors *gcl = new TGraphErrors();  
  // for (Double_t flux = fluxmin; flux<fluxmax; flux+=0.03)
  Int_t nabove=0; // number of simulated fluxes above requested C.L.
  Double_t flux = fluxmin;
  while (nabove<2)//  || (gcl->GetN()<10))
    {
      Int_t ncl=0;
      for (int i=0; i<n0; i++)
  	{
  	  fillhists(flux, dummyx, hofforg, kTRUE);
  	  Double_t min = scan_over_bins(ibin1, ibin2, kTRUE);
   	  if (min>min0)
   	    ncl++;
   	}
      Double_t p = 1.*ncl/n0;
      Double_t dp = sqrt(p*(1-p)/n0);
      gcl->SetPoint(gcl->GetN(), p, flux);
      gcl->SetPointError(gcl->GetN()-1, dp, 0.01);
      cout<<"flux = "<<flux<<", C.L="<<p<<"+-"<<dp<<endl;
      if (p>conf) nabove++;
      flux+=0.03;
    }
  
  gcl->Sort();
  if (!quiet)
    {
      TCanvas *ccl = new TCanvas ("ccl", "", 640, 480);
      gcl->Draw("A*L");
      gcl->GetXaxis()->SetTitle("C.L.");
      gcl->GetYaxis()->SetTitle("flux");
    }
  // Double_t xx[8];
  // gcl->LeastSquareFit(8, xx);
  // TF1 ffit("ffit", "pol7");
  // ffit.SetParameters(xx);
  // TF1 ffit("ffit", "1++x++x*x++pow(x,3)++pow(x,4)++pow(x,5)++pow(x,6)++pow(x,7)");
  // gcl->Fit(&ffit, "W");
  Double_t fluxlimit = gcl->Eval(conf);
  // gcl->Fit("pol7", "W");

  // Double_t fluxlimit = gcl->GetFunction("pol7")->Eval(conf);

  return fluxlimit;
}

void show_one_example(Double_t trueflux, Int_t seed, Bool_t singlebin=kFALSE)
{
  gRandom->SetSeed(seed);

  Double_t truex;
  fillhists(trueflux, truex);

  Double_t *rolkelimits = new Double_t[nbins];
  Double_t rolkelimit=calc_rolke(rolkelimits, kFALSE);

  TGraph *grolke = new TGraph();
  TGraph *grolke2 = new TGraph();
  grolke->SetMarkerStyle(32);
  grolke2->SetMarkerStyle(23);
  grolke2->SetMarkerColor(kBlue);
  for (int i=1; i<=nbins; i++)
    {
      grolke->SetPoint(grolke->GetN(), hon->GetBinCenter(i), rolkelimits[i-1]);
      if (rolkelimits[i-1] == rolkelimit)
	grolke2->SetPoint(grolke2->GetN(), hon->GetBinCenter(i), rolkelimits[i-1]);
    }
  delete[]rolkelimits;

  Double_t bayesmax, bayeslo, bayeshi, giacomoul;
  Double_t bayesul = bayes_scan(0, trueflux+3., bayesmax, bayeslo, bayeshi, giacomoul, kTRUE, singlebin);

  TLine *lbayes = new TLine (xmin, bayesul, xmax, bayesul);
  lbayes->SetLineWidth(3);
  lbayes->SetLineColor(kGreen);

  TH1F *honorg = (TH1F*) hon->Clone("honorg");
  TH1F *hofforg = (TH1F*) hoff->Clone("hofforg");
  TCanvas *cevents = new TCanvas ("cevents", "", 640, 480);
  cevents->SetTopMargin(0.01);
  cevents->SetRightMargin(0.02);
  cevents->SetLeftMargin(0.11);
  cevents->SetBottomMargin(0.12);
  honorg->GetXaxis()->SetTitleSize(0.06);
  honorg->GetXaxis()->SetTitleOffset(0.9);
  honorg->GetYaxis()->SetTitleSize(0.06);
  honorg->GetYaxis()->SetTitleOffset(0.9);
  honorg->GetXaxis()->SetLabelSize(0.06);
  honorg->GetYaxis()->SetLabelSize(0.06);
  honorg->Draw("E");
  hofforg->Draw("same");
  TLegend *leg1 = new TLegend (0.32, 0.14, 0.77, 0.28);
  leg1->SetNColumns(2);
  leg1->AddEntry(honorg, "ON region", "L");
  leg1->AddEntry(hofforg, "OFF region", "L");
  leg1->Draw();
  cevents->Print(Form("toy_events_flux%.2f.eps", trueflux));
  Double_t likelimit = likelimits(trueflux, hofforg, kFALSE); // this breaks hon and hoff histograms !

  TLine *lglobal = new TLine (xmin, likelimit, xmax, likelimit);
  lglobal->SetLineWidth(3);
  lglobal->SetLineStyle(kDashed);
  lglobal->SetLineColor(kRed);
  
  TCanvas *climits = new TCanvas ("climits", "", 640, 480);  
  climits->SetTopMargin(0.02);
  climits->SetRightMargin(0.02);
  climits->SetLeftMargin(0.11);
  climits->SetBottomMargin(0.12);
  TH1F *hosie = climits->DrawFrame(xmin, 0, xmax, trueflux+0.7);
  hosie->GetXaxis()->SetTitleSize(0.06);
  hosie->GetXaxis()->SetTitleOffset(0.9);
  hosie->GetYaxis()->SetTitleSize(0.06);
  hosie->GetYaxis()->SetTitleOffset(0.9);
  hosie->GetXaxis()->SetLabelSize(0.06);
  hosie->GetYaxis()->SetLabelSize(0.06);
  hosie->GetXaxis()->SetTitle("Position");
  hosie->GetYaxis()->SetTitle("Flux [a.u.]");
  grolke->Draw("P");
  grolke2->Draw("P");
  lglobal->Draw();
  lbayes->Draw();

  TGraph *gtrue = new TGraph ();
  gtrue->SetPoint(0, truex, trueflux);
  gtrue->SetMarkerSize(1.33);
  gtrue->Draw("*");
  TLegend *leg = new TLegend (0.3, 0.75, 0.6, 0.99);
  leg->AddEntry(grolke, "Individual limits", "P");
  leg->AddEntry(lglobal, "Frequentist limit", "L");
  leg->AddEntry(lbayes, "Bayesian limit", "L");
  leg->AddEntry(gtrue, "True flux", "P");
  leg->Draw();
  
  // climits->Print(Form("toy_limits_flux%.2f.eps", trueflux));

}

// ntf - number of fluxes to simulate from minflux to maxflux 
// npf - number of realizations per each flux
void compare_methods(Int_t ntf, Double_t minflux, Double_t maxflux, Int_t npf, Int_t seed=1)
{
  gRandom->SetSeed(seed);
  Double_t truefluxes[ntf];
  for (int i=0; i<ntf; i++)
    truefluxes[i]=minflux + (maxflux - minflux)*i/(ntf-1.);
  
  Double_t truex;
  // ofstream plikout ("out_stats.txt");
  ofstream plikout (Form("out_stats_seed%i_%ifluxes_%.2f_to_%.2f_%i_events.txt", seed, ntf, minflux, maxflux, npf));
  for (int itf=0; itf<ntf; itf++)
    for (int ipf=0; ipf<npf; ipf++)
      {
	fillhists(truefluxes[itf], truex);

	Double_t *rolkelimits = new Double_t[nbins];
	Double_t rolkelimit=calc_rolke(rolkelimits, kFALSE);

	Double_t bayesmax, bayeslo, bayeshi, giacomoul;
	Double_t bayesul = bayes_scan(0, truefluxes[itf]+3., bayesmax, bayeslo, bayeshi, giacomoul, kFALSE);
	
	TH1F *hofforg = (TH1F*) hoff->Clone("hofforg");
	// Double_t likelimit = likelimits(truefluxes[itf], hofforg); // this breaks hon and hoff histograms !
	Double_t likelimit=0;
	delete hofforg;

	cout<<"True="<<truefluxes[itf]<<", cons="<<rolkelimit<<", freq="<<likelimit<<", bayes="<<bayesul<<endl;
	plikout<<truefluxes[itf]<<" "<<rolkelimit<<" "<<likelimit<<" "<<bayesul<<" "<<giacomoul<<endl;
      }
  plikout.close();
}

void giacomos_test(Double_t trueflux, Int_t seed)
{
  gRandom->SetSeed(seed);

  Double_t truex;
  fillhists(trueflux, truex);

  Double_t bayesmax, bayeslo, bayeshi, giacomoul;
  Double_t bayesul = bayes_scan(0, trueflux+3., bayesmax, bayeslo, bayeshi, giacomoul, kTRUE);
}

void detection_bayes(Int_t seed, Double_t minflux, Int_t npf)
{
  gRandom->SetSeed(seed);
  ofstream plikout(Form("out_toy_bayes_p0_f%.2f_seed%i.txt", minflux, seed));
  TH1F *h = new TH1F ("h", "", 100, 0, 1);
  for (int iev=0; iev<npf; iev++)
    {
      Double_t truex;
      fillhists(minflux, truex);
      Double_t bayesmax, bayeslo, bayeshi, giacomoul;
      Double_t bayesul = bayes_scan(-1, 1., bayesmax, bayeslo, bayeshi, giacomoul, kFALSE);
      
      Int_t iclosest=-1;
      Double_t diffclosest=1.e9;
      for (int i=1; i<=ghprob->GetNbinsX(); i++)
	{
	  Double_t diff = TMath::Abs(ghprob->GetBinLowEdge(i+1));
	  if (diff<diffclosest)
	    {
	      diffclosest=diff;
	      iclosest =i;
	    }
	}
      Double_t psum=ghprob->Integral(0, iclosest);
      cout<<iev<<" closest bin="<<iclosest<<", sum = "<<psum<<endl;
      plikout<<psum<<endl;
      h->Fill(psum);
    }
  plikout.close();
  TCanvas *cpsum = new TCanvas ("cpsum", "", 640, 480);
  h->Draw();
}

void detection_freq(Int_t seed, Double_t minflux, Int_t npf)
{
  gRandom->SetSeed(seed);
  ofstream plikout(Form("out_toy_freq_p0_f%.2f_seed%i.txt", minflux, seed));
  TH1F *h = new TH1F ("h", "", 1000, -7, 2);
  Int_t ibin1=1, ibin2=nbins; // full range
  for (int iev=0; iev<npf; iev++)
    {
      if (iev% 10000 == 1) cout<<iev<<"/"<<npf<<endl;
      Double_t truex;
      fillhists(minflux, truex);
      Double_t min = scan_over_bins(ibin1, ibin2, kTRUE);
      h->Fill(min);
      plikout<<min<<endl;
    }
  plikout.close();
  TCanvas *cpsum = new TCanvas ("cpsum", "", 640, 480);
  h->Draw();
}

void detection_agnostic(Int_t seed, Double_t minflux, Int_t npf)
{
  gRandom->SetSeed(seed);
  ofstream plikout(Form("out_toy_agnostic_p0_f%.2f_seed%i.txt", minflux, seed));
  TH1F *h = new TH1F ("h", "", 1000, -5, 5);
  Int_t ibin1=1, ibin2=nbins; // full range
  for (int iev=0; iev<npf; iev++)
    {
      if (iev% 10000 == 1) cout<<iev<<"/"<<npf<<endl;
      Double_t truex;
      fillhists(minflux, truex);

      Int_t bestbin=-1;
      Double_t bestsigma=-1.e9;
      for (int ibin = 1; ibin<=nbins; ibin++)
	{
	  Double_t on = hon->GetBinContent(ibin);
	  Double_t offall=offmulti * hoff->GetBinContent(ibin);
	  Double_t sigma = MMath::SignificanceLiMaSigned(on, offall, 1/offmulti);
	  if (sigma>bestsigma)
	    {
	      bestsigma=sigma;
	      bestbin=ibin;
	    }
	}
      Double_t probtrial = MMath::ProbOneOutOfMany(MMath::ProbSigma(bestsigma)/2, nbins);
      h->Fill(bestsigma);
      plikout<<bestsigma<<" "<<probtrial<<endl;
    }
  plikout.close();
  TCanvas *cpsum = new TCanvas ("cpsum", "", 640, 480);
  h->Draw();
}

void test_agnostic(Int_t seed=1, Double_t trueflux=2.5)
{
  gRandom->SetSeed(seed);
  TH1F *h = new TH1F ("h", "", 125, 0, 3.5);
  TH1F *h2 = new TH1F ("h2", ";position;limit position", nbins, xmin, xmax);
  TH1F *h3 = new TH1F ("h3", "", 125, 0, 3.5);
  Int_t npf=10000;
  Int_t nok=0;
  Int_t nmiss=0;
  for (int iev=0; iev<npf; iev++)
    {
      if (iev% 10000 == 1) cout<<iev<<"/"<<npf<<endl;
      Double_t truex;
      fillhists(trueflux, truex);
      Double_t *rolkelimits = new Double_t[nbins];
      Double_t rolkelimit=calc_rolke(rolkelimits, kFALSE);
      if (rolkelimit>trueflux)
	nok++;
      Int_t ibin=hon->FindBin(truex);
      for (int i=0; i<nbins; i++)
	if(rolkelimits[i]==rolkelimit)
	  {
	    h2->Fill(h2->GetBinCenter(i+1));
	    if (ibin!=i+1)
	      nmiss++;
	  }
      h->Fill(rolkelimit);
      h3->Fill((hon->GetBinContent(ibin)-hoff->GetBinContent(ibin))/haeff->GetBinContent(ibin));
    }
  cout<<"nmiss="<<nmiss<<endl;
  cout<<"C.L.="<<100.*nok/npf<<"%"<<endl;
  TCanvas *cpsum = new TCanvas ("cpsum", "", 640, 480);
  h->Draw();
  TCanvas *c2 = new TCanvas ("c2", "", 640, 480);
  h2->Draw();
  TCanvas *c3 = new TCanvas ("c3", "", 640, 480);
  h3->Draw();
  
}

// this used the probability distribution obtained in the bayesian method
// to get the maximum, and treat it in a frequentist way
void  bayesian_freq(Int_t seed, Double_t f0min=0, Double_t f0max=0)
{
  gRandom->SetSeed(seed);
  Int_t nf0=200; //number of bins to scan the flux 
  nf0/=2;
  // first generate a lot of flux=0 realizations
  Int_t n0=5000;// number of noise-only
  Double_t dummyx=0;

  Double_t flux = f0min;
  TGraph *gprob_median = new TGraph();
  TGraph *gprob_cl = new TGraph();
  TGraph *gts_median = new TGraph();
  TGraph *gts_cl = new TGraph();
  TH2D *hprob = new TH2D ("hprob", "Probability;Position;Flux", nbins, xmin, xmax, nf0, f0min, f0max+0.7);
  while (flux<f0max)
    {
      TH1F *hflux0= new TH1F ("hflux0", "Flux = 0;best flux;N", nf0*10, f0min, f0max+0.7);
      TH1F *hts0= new TH1F ("hts0", "Flux = 0;best TS;N", 1500, -10, 70);
      for (int i=0; i<n0; i++)
	{
	  if (i%100 ==1 ) cout<<i<<" "<<n0<<"="<<100.*i/n0<<"% done"<<endl;
	  fillhists(flux, dummyx, 0, kTRUE);
	  Double_t bestflux=-1, bestprob=-1;
	  for (int i1=1; i1<=nbins; i1++)
	    for (int i2=1; i2<=nf0; i2++)
	      {
		Double_t x0=hprob->GetXaxis()->GetBinCenter(i1);
		Double_t f0=hprob->GetYaxis()->GetBinCenter(i2);
		Double_t pthis=p_d_true(x0, f0);
		if (pthis>bestprob)
		  {
		    bestflux=f0;
		    bestprob=pthis;
		  }
	      }
	  hflux0->Fill(bestflux);
	  
	  Double_t bestts = scan_over_bins(0, nbins, kTRUE);
	  hts0->Fill(bestts);
	}
      hflux0->ComputeIntegral();
      Double_t q=0.5, median_prob=-1, fluxul_prob=-1, confm1=1-conf;
      hflux0->GetQuantiles(1, &median_prob, &q);
      hflux0->GetQuantiles(1, &fluxul_prob, &confm1);
      cout<<"true flux = "<<flux<<", max likelihood flux median="<<median_prob<<", "<<conf<<" C.L. limit <"<<fluxul_prob<<", mean = "<<hflux0->GetMean()<<" under/overflow = "<<hflux0->GetBinContent(0)<<" "<<hflux0->GetBinContent(hflux0->GetNbinsX()+1)<<endl;
      gprob_median->SetPoint(gprob_median->GetN(), median_prob, flux);
      gprob_cl->SetPoint(gprob_cl->GetN(), fluxul_prob, flux);

      Double_t median_ts=-1, fluxul_ts=-1;
      hts0->GetQuantiles(1, &median_ts, &q);
      hts0->GetQuantiles(1, &fluxul_ts, &confm1);
      cout<<"true flux = "<<flux<<", max likelihood ts median="<<median_ts<<", "<<conf<<" C.L. limit <"<<fluxul_ts<<", mean = "<<hts0->GetMean()<<" under/overflow = "<<hts0->GetBinContent(0)<<" "<<hts0->GetBinContent(hts0->GetNbinsX()+1)<<endl;
      gts_median->SetPoint(gts_median->GetN(), median_ts, flux);
      gts_cl->SetPoint(gts_cl->GetN(), fluxul_ts, flux);

      delete hflux0;
      delete hts0;
      flux+=0.03;
    }
  delete hprob;

  gprob_median->Sort();
  gprob_cl->Sort();
      // Double_t p = 1.*ncl/n0;
      // Double_t dp = sqrt(p*(1-p)/n0);

  TCanvas *cc1 = new TCanvas ("cc1", "", 640, 480);
  gprob_median->Draw("A*L");
  TCanvas *cc2 = new TCanvas ("cc2", "", 640, 480);
  gprob_cl->Draw("A*L");
  TCanvas *cc3 = new TCanvas ("cc3", "", 640, 480);
  gts_median->Draw("A*L");
  TCanvas *cc4 = new TCanvas ("cc4", "", 640, 480);
  gts_cl->Draw("A*L");

  TFile *plikout = new TFile("bayes_freq_prob_ts.root", "RECREATE");
  plikout->cd();
  gprob_median->Write("gprob_median");
  gprob_cl->Write("gprob_cl");
  gts_median->Write("gts_median");
  gts_cl->Write("gts_cl");
  plikout->Close();
}

void bayesian_freq_read()
{
  TH2F *h0 = new TH2F ("h0", ";limit (TS method); limit (prob method)", 100, 0.4, 1, 100, 0.4, 1);

  TFile *plikin = new TFile("bayes_freq_prob_ts_seed1_5000ev.root");
  // TFile *plikin = new TFile("bayes_freq_prob_ts.root");
  TGraph *gprob_median = (TGraph *) plikin->Get("gprob_median");
  TGraph *gprob_cl = (TGraph *) plikin->Get("gprob_cl");
  TGraph *gts_median = (TGraph *) plikin->Get("gts_median");
  TGraph *gts_cl = (TGraph *) plikin->Get("gts_cl");

  TCanvas *c1 = new TCanvas ("c1", "", 640, 480);
  gprob_cl->Draw("A*L");
  TCanvas *c2 = new TCanvas ("c2", "", 640, 480);
  gts_cl->Draw("A*L");
  
  Int_t nf0=200/2; //number of bins to scan the flux
  Double_t f0min=0, f0max=1;
  TH2D *hprob = new TH2D ("hprob", "Probability;Position;Flux", nbins, xmin, xmax, nf0, f0min, f0max);
  

  Double_t ts_ul0 = -1; // U.L. for TS method with flux = 0;
  Double_t prob_ul0 = -1; // U.L. for prob method with flux = 0;
  for (int i=0; i<gprob_median->GetN(); i++)
    if (gprob_median->GetY()[i]<1.e-6)
      prob_ul0=gprob_median->GetX()[i];
  for (int i=0; i<gts_median->GetN(); i++)
    if (gts_median->GetY()[i]<1.e-6)
      ts_ul0=gts_median->GetX()[i];
  cout<<"U.L. for 0 flux, TS method="<<ts_ul0<<", prob method="<<prob_ul0<<endl;

  gRandom->SetSeed(2);
  Int_t n0=400;
  Double_t flux=0.5, dummyx=-1;
  for (int i=0; i<n0; i++)
    {
      if (i%100 ==1 ) cout<<i<<" "<<n0<<"="<<100.*i/n0<<"% done"<<endl;
      fillhists(flux, dummyx, 0, kTRUE);
      Double_t bestflux=-1, bestprob=-1;
      for (int i1=1; i1<=nbins; i1++)
	for (int i2=1; i2<=nf0; i2++)
	  {
	    Double_t x0=hprob->GetXaxis()->GetBinCenter(i1);
	    Double_t f0=hprob->GetYaxis()->GetBinCenter(i2);
	    Double_t pthis=p_d_true(x0, f0);
	    if (pthis>bestprob)
	      {
		bestflux=f0;
		bestprob=pthis;
	      }
	  }
      if (bestflux<prob_ul0)
	bestflux=prob_ul0;
      
      Double_t ul_prob = gprob_cl->Eval(bestflux);
      
      Double_t bestts = scan_over_bins(0, nbins, kTRUE);
      if (bestts<ts_ul0)
	bestts=ts_ul0;
      Double_t ul_ts = gts_cl->Eval(bestts);

      cout<<(bestflux==prob_ul0?"*":" ")<<(bestts==ts_ul0?"*":" ");
      cout<<" prob method: "<<ul_prob<<" ( from "<<bestflux<<"), TS method: "<<ul_ts<<" ( from "<<bestts<<")"<< endl;
      h0->Fill(ul_ts, ul_prob);
    }

  plikin->Close();
  TCanvas *cc = new TCanvas ("cc", "", 640, 640);
  TLine *l = new TLine(0.4, 0.4, 1, 1);
  h0->Draw("colz");
  l->Draw();
}

void appendixa()
{
  show_one_example(2., 6, kTRUE);
  TH1D *hsingle = (TH1D*) hhh->Clone("hsingle");
  show_one_example(2., 6, kFALSE);
  TH1D *hall = (TH1D*) hhh->Clone("hall");

  TCanvas *ccc = new TCanvas ("ccc", "", 640, 480);
  ccc->SetTopMargin(0.01);
  ccc->SetRightMargin(0.01);
  ccc->SetLogy();
  hall->SetLineColor(kRed);
  hsingle->SetTitle("");
  hall->SetTitle("");
  hsingle->SetStats(0);
  hall->SetStats(0);
  hsingle->SetLineStyle(kDotted);
  hsingle->Draw("HIST X");
  hall->Draw("HIST X same");
}

// to avoid the problem with negative upper limits we compute the C.L. that the flux is > 0
void freq_bayes_calccl(Double_t trueflux=0, Int_t npf=0, Int_t seed=1)
{
  gRandom->SetSeed(seed);
  Double_t dummyx=0;

  ofstream plikout(Form("out_freq_bayes_calccl_p0_f%.2f_seed%i.txt", trueflux, seed));
  for (int ipf=0; ipf<npf; ipf++)
    {
      fillhists(trueflux, dummyx);
      Double_t min = scan_over_bins(1, nbins, kTRUE);
      TH1F *hofforg = (TH1F*) hoff->Clone("hofforg");
      
      Double_t bayesmax, bayeslo, bayeshi, giacomoul;
      Double_t bayesul = bayes_scan(-1., trueflux+1., bayesmax, bayeslo, bayeshi, giacomoul, kFALSE);
      Double_t clbayes=1;
      Int_t ibin=1;
      while (hhh->GetBinCenter(ibin)<0)
	{
	  clbayes-=hhh->GetBinContent(ibin);
	  ibin++;
	}
      Int_t n0=1000; // number of realizations with 0 flux
      Int_t nbelow=0;
      for (int i=0; i<n0; i++)
	{
	  fillhists(0, dummyx, hofforg, kTRUE);
	  Double_t min0 = scan_over_bins(1, nbins, kTRUE);
	  if (min0<min)
	    nbelow++;
	}
      delete hofforg;
      cout<<"C.L.="<<100.*nbelow/n0<<" "<<clbayes*100.<<endl;
      plikout<<1.*nbelow/n0<<" "<<clbayes<<endl;
    }
  plikout.close();

}

// testing some things with frequentist vs agnostic method
void test_freq(Double_t trueflux=0, Int_t seed=1)
{
  gRandom->SetSeed(seed);

  Double_t truex;
  // make a table with standarized fluxes 
  TGraph *gcl = new TGraph();  
  Int_t n0=4000;
  for (Double_t flux = trueflux; flux< trueflux +1.2; flux+=0.03)
    {
      Int_t ncl=0;
      TH1F *h = new TH1F("h", "", 10000, -10, 250);
      for (int i=0; i<n0; i++)
  	{
  	  fillhists(flux, truex, 0, kTRUE);
  	  Double_t min = scan_over_bins(1, nbins, kTRUE);
	  h->Fill(min);
   	}
      Double_t f=1-conf, quantile;
      h->ComputeIntegral();
      h->GetQuantiles(1, &quantile, &f);
      cout<<flux<<" "<<quantile<<" "<<h->GetBinContent(0)<<" "<<h->GetBinContent(h->GetNbinsX()+1)<<endl;
      gcl->SetPoint(gcl->GetN(), quantile, flux);
      delete h;
    }
  gcl->Sort();
  TCanvas *c = new TCanvas ("c", "", 640, 480);
  gcl->Draw("A*L");

  TH2F *h2 = new TH2F ("h2", ";agnostic;frequentist", 100, 0, 2, 100, 0, 2);
  TH2F *h3 = new TH2F ("h3", ";exposure;frequentist/agnostic", 100, 0, 105, 100, 0, 3);
  
  for (int i=0; i<10000; i++)
    {
      fillhists(trueflux, truex);      
      
      Double_t *rolkelimits = new Double_t[nbins];
      Double_t rolkelimit=calc_rolke(rolkelimits, kFALSE);
      Double_t min = scan_over_bins(1, nbins, kTRUE);
      Double_t freqlimit=gcl->Eval(min);
      h2->Fill(rolkelimit, freqlimit);
      h3->Fill(haeff->GetBinContent(haeff->FindBin(truex)), freqlimit/rolkelimit);
    }
  TCanvas *c2 = new TCanvas ("c2", "", 640, 480);
  h2->Draw("colz");
  TCanvas *c3 = new TCanvas ("c3", "", 640, 480);
  h3->Draw("colz");
}

void toy_paper(Int_t ntf=0, Double_t minflux=0, Double_t maxflux=0, Int_t npf=0, Int_t seed=1)
{
  prep_irf();
  cout<<ntf<<" "<<minflux<<" "<<maxflux<<" "<<npf<<" "<<seed<<endl;
  // test_agnostic();
  // giacomos_test(2., 21);
  // show_one_example(0, 2);
  // show_one_example(0.3, 4); 
  // show_one_example(0.45, 5);
  show_one_example(1.5, 5);
  // appendixa();
  // freq_bayes_calccl(0.5, 10, 1);
  // show_one_example(0.5, 5);
 //   compare_methods(16,  0.5, 2., 250);
  // compare_methods(ntf, minflux, maxflux, npf, seed);
  // detection_bayes(seed, minflux, npf);
  // detection_freq(seed, minflux, npf);
  // detection_agnostic(seed, minflux, npf);
  // bayesian_freq(1, 0, 1.);
  // bayesian_freq_read();
  // test_freq(1., 25);
}
