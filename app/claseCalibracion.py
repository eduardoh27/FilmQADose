import numpy as np
from PIL import Image
import SimpleITK as sitk
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar,minimize
from imagenMatplotlibLibre import *
from scipy.stats.distributions import chi2
import dill
import pickle


def funcionRacionalLineal(x,p0,p1,p2):
    return -np.log10((p0+p1*x)/(x+p2))

def funcionInversaRacionalLineal(y,p0,p1,p2):
    r=10**-y
    return (p0-p2*(r))/((r-p1))
    
def funcionRacionalLinealMulti(x,p0,p1,p2):
    return (p0+p1*x)/(x+p2)

def funcionInversaRacionalLinealMulti(y,p0,p1,p2):
    return (p0-p2*y)/(y-p1)

def derivadaInversaRacionalLinealMulti(y,p0,p1,p2):
    return (p2*p1-p0)/((y-p1)**2)
    
def derivada2InversaRacionalLinealMulti(y,p0,p1,p2):
    return 2*(p0-p1*p2)/((y-p1)**3)
    
def funcionCubica(x,p0,p1,p2,p3):
    return p0+p1*x+p2*x*x+p2*x*x*x
    
def funcionExponencialPolinomica(x,p0,p1,p2):
    return  p0+p1*(x**p2)
    
def funcionLineal(x,p0,p1):
    return  p0+p1*(x)   
    

def guardar_calibracion(tipoCanal,tipoCurva,parametros,covarianzas,dosis,transmitancias,ceros,incertidumbres,funcionesRGB,funcionCalDel,funcionTaD,labels,nombreArchivo):
    f=open(nombreArchivo,'a')
    dis={
        "Nombre": nombreArchivo,
        "TipoCanal": tipoCanal,
        "TipoCurva": tipoCurva,
        "Parametros": parametros,
        "Covarianzas": covarianzas,
        "Dosis": dosis,
        "Dopticas": transmitancias,
        "Ceros": ceros,
        "Incertidumbres": incertidumbres,
        "funcionesRGB": funcionesRGB,
        "funcionCalDel": funcionCalDel,
        "labels": labels,
        "funcionTaD": funcionTaD}
    dill.settings['recurse'] = True
    dill.dump(dis, open(nombreArchivo, 'wb'))

def guardar_calibracion0(tipoCanal, tipoCurva, parametros, covarianzas, dosis, transmitancias, ceros, incertidumbres, funcionesRGB, funcionCalDel, funcionTaD, labels, nombreArchivo):
    dis = {
        "Nombre": nombreArchivo,
        "TipoCanal": tipoCanal,
        "TipoCurva": tipoCurva,
        "Parametros": parametros,
        "Covarianzas": covarianzas,
        "Dosis": dosis,
        "Dopticas": transmitancias,
        "Ceros": ceros,
        "Incertidumbres": incertidumbres,
        "funcionesRGB": funcionesRGB,
        "funcionCalDel": funcionCalDel,
        "labels": labels,
        "funcionTaD": funcionTaD
    }
    # Serializar con pickle
    with open(nombreArchivo, 'wb') as f:
        pickle.dump(dis, f)
    


def leer_Calibracion(nombreArchivo):
    print('PICKLE')
    with open(nombreArchivo, 'rb') as f:
        return pickle.load(f)


def leer_Calibracion0(nombreArchivo):
    f=open(nombreArchivo,'rb')
    return dill.load(f)
    
class CalibracionImagen:
    def __init__(self,promsR,promsG,promsB,incertidumbres,dosis,tipoCanal,tipoCurva):
        self.netODR=promsR[0]
        self.netODG=promsG[0]
        self.netODB=promsB[0]
        
        self.promsRCero=promsR[1]
        self.promsGCero=promsG[1]
        self.promsBCero=promsB[1]
        
        self.promsR=promsR[2]
        self.promsG=promsG[2]
        self.promsB=promsB[2]
        
        self.stdR=incertidumbres[1][0]
        self.stdG=incertidumbres[1][1]
        self.stdB=incertidumbres[1][2]
        
        self.stdRCero=incertidumbres[2][0]
        self.stdGCero=incertidumbres[2][1]
        self.stdBCero=incertidumbres[2][2]
        
        self.varODR=incertidumbres[0][0]
        self.varODG=incertidumbres[0][1]
        self.varODB=incertidumbres[0][2]


        self.dosis=dosis
        self.tipoCanal=tipoCanal
        self.tipoCurva=tipoCurva
        self.erroresR=[]
        self.erroresG=[]
        self.erroresB=[]
        self.parametrosOptimosR=[]
        self.parametrosOptimosG=[]
        self.parametrosOptimosB=[]
        self.pCovarianzaR=[]
        self.pCovarianzaG=[]
        self.pCovarianzaB=[]
        
        self.labelFuncion=''
        self.labelR=''
        self.labelG=''
        self.labelB=''
        def f():
            return 1
        self.funcionTaD=f
        self.funcionCalculaDelta=f
        
    def generar_calibracion(self,nombreArchivo):
        def fR(D):
            return 0
        def fG(D):
            return 0
        def fB(D):
            return 0
        def fRder(D):
            return 0
        def fGder(D):
            return 0
        def fBder(D):
            return 0
        def fRder2(D):
            return 0
        def fGder2(D):
            return 0
        def fBder2(D):
            return 0
        if self.tipoCanal=="Multicanal":
            cambioR=self.promsRCero-self.promsRCero[0]
            cambioG=self.promsGCero-self.promsGCero[0]
            cambioB=self.promsBCero-self.promsBCero[0]
            self.promsR=self.promsR+cambioR
            self.promsG=self.promsG+cambioG
            self.promsB=self.promsB+cambioB
            self.parametrosOptimosR,self.pCovarianzaR=curve_fit(funcionRacionalLinealMulti, self.dosis, self.promsR)
            self.parametrosOptimosG,self.pCovarianzaG=curve_fit(funcionRacionalLinealMulti, self.dosis, self.promsG)
            self.parametrosOptimosB,self.pCovarianzaB=curve_fit(funcionRacionalLinealMulti, self.dosis, self.promsB)
            
            self.pCovarianzaR=np.sqrt(np.diag(self.pCovarianzaR))
            self.pCovarianzaG=np.sqrt(np.diag(self.pCovarianzaG))
            self.pCovarianzaB=np.sqrt(np.diag(self.pCovarianzaB))
            def fRa(oD):
                return funcionInversaRacionalLinealMulti(oD,*self.parametrosOptimosR)
            def fGa(oD):
                return funcionInversaRacionalLinealMulti(oD,*self.parametrosOptimosG)
            def fBa(oD):
                return funcionInversaRacionalLinealMulti(oD,*self.parametrosOptimosB)
            def fRdera(oD):
                return derivadaInversaRacionalLinealMulti(oD,*self.parametrosOptimosR)
            def fGdera(oD):
                return derivadaInversaRacionalLinealMulti(oD,*self.parametrosOptimosG)
            def fBdera(oD):
                return derivadaInversaRacionalLinealMulti(oD,*self.parametrosOptimosB)
            def fRder2a(oD):
                return derivada2InversaRacionalLinealMulti(oD,*self.parametrosOptimosR)
            def fGder2a(oD):
                return derivada2InversaRacionalLinealMulti(oD,*self.parametrosOptimosG)
            def fBder2a(oD):
                return derivada2InversaRacionalLinealMulti(oD,*self.parametrosOptimosB)   
            fR=fRa
            fG=fGa
            fB=fBa
            fRder=fRdera
            fGder=fGdera
            fBder=fBdera
            fRder2=fRder2a
            fGder2=fGder2a
            fBder2=fBder2a     
            self.erroresR=np.sqrt(self.stdR**2+self.stdRCero**2)
            self.erroresG=np.sqrt(self.stdG**2+self.stdGCero**2)
            self.erroresB=np.sqrt(self.stdB**2+self.stdBCero**2)
            self.labelFuncion=r'$T=\frac{a+bD}{c+D}$'
            self.labelR=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[0],self.pCovarianzaR[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[1],self.pCovarianzaR[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[2],self.pCovarianzaR[2])
            self.labelG=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[0],self.pCovarianzaG[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[1],self.pCovarianzaG[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[2],self.pCovarianzaG[2])
            self.labelB=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[0],self.pCovarianzaB[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[1],self.pCovarianzaB[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[2],self.pCovarianzaB[2])
            
        elif self.tipoCurva=="Racional lineal":
            
            self.parametrosOptimosR,self.pCovarianzaR=curve_fit(funcionRacionalLineal, self.dosis, self.netODR,sigma=self.varODR)
            self.parametrosOptimosG,self.pCovarianzaG=curve_fit(funcionRacionalLineal, self.dosis, self.netODG,sigma=self.varODG)
            self.parametrosOptimosB,self.pCovarianzaB=curve_fit(funcionRacionalLineal, self.dosis, self.netODB,sigma=self.varODB)
            
            self.pCovarianzaR=np.sqrt(np.diag(self.pCovarianzaR))
            self.pCovarianzaG=np.sqrt(np.diag(self.pCovarianzaG))
            self.pCovarianzaB=np.sqrt(np.diag(self.pCovarianzaB))
            

            def fRa(oD):
                return funcionInversaRacionalLineal(oD,*self.parametrosOptimosR)
            def fGa(oD):
                return funcionInversaRacionalLineal(oD,*self.parametrosOptimosG)
            def fBa(oD):
                return funcionInversaRacionalLineal(oD,*self.parametrosOptimosB)
            fR=fRa
            fG=fGa
            fB=fBa
            self.erroresR=self.varODR
            self.erroresG=self.varODG
            self.erroresB=self.varODB
            self.labelFuncion=r'$oD=-log_{10}\left(\frac{a+bD}{c+D}\right)$'
            self.labelR=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[0],self.pCovarianzaR[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[1],self.pCovarianzaR[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[2],self.pCovarianzaR[2])
            self.labelG=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[0],self.pCovarianzaG[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[1],self.pCovarianzaG[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[2],self.pCovarianzaG[2])
            self.labelB=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[0],self.pCovarianzaB[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[1],self.pCovarianzaB[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[2],self.pCovarianzaB[2])

            
        elif self.tipoCurva=="Cubica":
            self.parametrosOptimosR,self.pCovarianzaR=curve_fit(funcionCubica, self.netODR, self.dosis,sigma=self.varODR)
            self.parametrosOptimosG,self.pCovarianzaG=curve_fit(funcionCubica, self.netODG, self.dosis,sigma=self.varODG)
            self.parametrosOptimosB,self.pCovarianzaB=curve_fit(funcionCubica, self.netODB, self.dosis,sigma=self.varODB)
            
            self.pCovarianzaR=np.sqrt(np.diag(self.pCovarianzaR))
            self.pCovarianzaG=np.sqrt(np.diag(self.pCovarianzaG))
            self.pCovarianzaB=np.sqrt(np.diag(self.pCovarianzaB))
            def fRa(oD):
                return funcionCubica(oD,*self.parametrosOptimosR)
            def fGa(oD):
                return funcionCubica(oD,*self.parametrosOptimosG)
            def fBa(oD):
                return funcionCubica(oD,*self.parametrosOptimosB)
            fR=fRa
            fG=fGa
            fB=fBa
            self.erroresR=self.varODR
            self.erroresG=self.varODG
            self.erroresB=self.varODB
            self.labelFuncion=r'$D=ax^3+bx^2+cx+d$'
            self.labelR=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[0],self.pCovarianzaR[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[1],self.pCovarianzaR[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[2],self.pCovarianzaR[2])+'\n'+r"$d={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[3],self.pCovarianzaR[3])
            self.labelG=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[0],self.pCovarianzaG[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[1],self.pCovarianzaG[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[2],self.pCovarianzaG[2])+'\n'+r"$d={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[3],self.pCovarianzaG[3])
            self.labelB=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[0],self.pCovarianzaB[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[1],self.pCovarianzaB[1])+'\n'+r"$c={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[2],self.pCovarianzaB[2])+'\n'+r"$d={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[3],self.pCovarianzaB[3])
            
        elif self.tipoCurva=="Exponencial polinomica":
            self.parametrosOptimosR,self.pCovarianzaR=curve_fit(funcionExponencialPolinomica, self.netODR, self.dosis,sigma=self.varODR)
            self.parametrosOptimosG,self.pCovarianzaG=curve_fit(funcionExponencialPolinomica, self.netODG, self.dosis,sigma=self.varODG)
            self.parametrosOptimosB,self.pCovarianzaB=curve_fit(funcionExponencialPolinomica, self.netODB, self.dosis,sigma=self.varODB)
            self.pCovarianzaR=np.sqrt(np.diag(self.pCovarianzaR))
            self.pCovarianzaG=np.sqrt(np.diag(self.pCovarianzaG))
            self.pCovarianzaB=np.sqrt(np.diag(self.pCovarianzaB))
            def fRa(oD):
                return funcionExponencialPolinomica(oD,*self.parametrosOptimosR)
            def fGa(oD):
                return funcionExponencialPolinomica(oD,*self.parametrosOptimosG)
            def fBa(oD):
                return funcionExponencialPolinomica(oD,*self.parametrosOptimosB)
            fR=fRa
            fG=fGa
            fB=fBa
            self.erroresR=self.varODR
            self.erroresG=self.varODG
            self.erroresB=self.varODB
            self.labelFuncion=r'$D=a+b(oD)^n$'
            self.labelR=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[0],self.pCovarianzaR[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[1],self.pCovarianzaR[1])+'\n'+r"$n={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[2],self.pCovarianzaR[2])
            self.labelG=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[0],self.pCovarianzaG[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[1],self.pCovarianzaG[1])+'\n'+r"$n={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[2],self.pCovarianzaG[2])
            self.labelB=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[0],self.pCovarianzaB[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[1],self.pCovarianzaB[1])+'\n'+r"$n={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[2],self.pCovarianzaB[2])

            
        elif self.tipoCurva=="Lineal":
            self.parametrosOptimosR,self.pCovarianzaR=curve_fit(funcionLineal, self.netODR, self.dosis,sigma=self.varODR)
            self.parametrosOptimosG,self.pCovarianzaG=curve_fit(funcionLineal, self.netODG, self.dosis,sigma=self.varODG)
            self.parametrosOptimosB,self.pCovarianzaB=curve_fit(funcionLineal, self.netODB, self.dosis,sigma=self.varODB)
            self.pCovarianzaR=np.sqrt(np.diag(self.pCovarianzaR))
            self.pCovarianzaG=np.sqrt(np.diag(self.pCovarianzaG))
            self.pCovarianzaB=np.sqrt(np.diag(self.pCovarianzaB))
            def fRa(oD):
                return funcionLineal(oD,*self.parametrosOptimosR)
            def fGa(oD):
                return funcionLineal(oD,*self.parametrosOptimosG)
            def fBa(oD):
                return funcionLineal(oD,*self.parametrosOptimosB)
            fR=fRa
            fG=fGa
            fB=fBa
            self.erroresR=self.varODR
            self.erroresG=self.varODG
            self.erroresB=self.varODB
            self.labelFuncion=r'$D=a*(oD)+b$'
            self.labelR=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[0],self.pCovarianzaR[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosR[1],self.pCovarianzaR[1])
            self.labelG=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[0],self.pCovarianzaG[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosG[1],self.pCovarianzaG[1])
            self.labelB=r"$a={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[0],self.pCovarianzaB[0])+'\n'+r"$b={0:.3g} \pm {1:.3g}$".format(self.parametrosOptimosB[1],self.pCovarianzaB[1])
                

        if self.tipoCanal=="Canal solo":
            def far(odR,odG,odB):
                return np.dstack((fR(odR),fG(odG),fB(odB)))
            self.funcionTaD=far
            
        else:
            def fMickDiferencia(DeltaD,oDR,oDG,oDB):
                a=fR(oDR*DeltaD)
                b=fG(oDG*DeltaD)
                c=fB(oDB*DeltaD)
                return (a-b)**2+(a-c)**2+(b-c)**2
                
            def fMickDiferenciaDerivada(DeltaD,oDR,oDG,oDB):
                a=fR(oDR*DeltaD)
                b=fG(oDG*DeltaD)
                c=fB(oDB*DeltaD)
                ad=oDR*fRder(oDR*DeltaD)
                bd=oDG*fGder(oDG*DeltaD)
                cd=oDB*fBder(oDB*DeltaD)
                return 2*((ad-bd)*(a-b)+(ad-cd)*(a-c)+(bd-cd)*(b-c))
            def fMickDiferenciaDerivada2(DeltaD,oDR,oDG,oDB):
                a=fR(oDR*DeltaD)
                b=fG(oDG*DeltaD)
                c=fB(oDB*DeltaD)
                ad=oDR*fRder(oDR*DeltaD)
                bd=oDG*fGder(oDG*DeltaD)
                cd=oDB*fBder(oDB*DeltaD)
                ad2=(oDR**2)*fRder2(oDR*DeltaD)
                bd2=(oDG**2)*fGder2(oDG*DeltaD)
                cd2=(oDB**2)*fBder2(oDB*DeltaD)
                return 2*((ad2-bd2)*(a-b)+(ad-bd)**2+(ad2-cd2)*(a-c)+(ad-cd)**2+(bd2-cd2)*(b-c)+(bd-cd)**2)
              
            def deltaOptimo(oDR,oDG,oDB):
                x0=0*oDR+1
                N=20
                for i in range(N):
                    x0=x0-fMickDiferenciaDerivada(x0,oDR,oDG,oDB)/fMickDiferenciaDerivada2(x0,oDR,oDG,oDB)
                    print(i)
                return x0
                
            def fux(DeltaD,oDR,oDG,oDB):
                return np.dstack((fR(DeltaD*oDR),fG(DeltaD*oDG),fB(DeltaD*oDB)))
                
            self.funcionCalculaDelta=deltaOptimo
            self.funcionTaD=fux
            

                             
                     
        grafica=ImagenMatplotlibLibre(None,style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        grafica.ax.grid()
        grafica.ax.set_xlabel("Dosis(Gy)")
        



        
        if self.tipoCanal!="Multicanal":
            grafica.ax.set_ylabel("Densidad Óptica")
            grafica.ax.text(0, 0.4, self.labelFuncion, color='black', 
                    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            grafica.ax.errorbar(self.dosis,self.netODR,yerr=self.erroresR,color='r',fmt='o',markersize=2)
            grafica.ax.errorbar(self.dosis,self.netODG,yerr=self.erroresG,color='g',fmt='o',markersize=2)
            grafica.ax.errorbar(self.dosis,self.netODB,yerr=self.erroresB,color='b',fmt='o',markersize=2)
            xasR=np.linspace(self.netODR[0],self.netODR[-1]+0.005,100)
            xasG=np.linspace(self.netODG[0],self.netODG[-1]+0.005,100)
            xasB=np.linspace(self.netODB[0],self.netODB[-1]+0.005,100)
        
        else: 
            grafica.ax.set_ylabel("Transmitancia")
            grafica.ax.text(8.2, 0.65, self.labelFuncion, color='black', 
                    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            grafica.ax.errorbar(self.dosis,self.promsR,yerr=self.erroresR,color='r',fmt='o',markersize=2)
            grafica.ax.errorbar(self.dosis,self.promsG,yerr=self.erroresG,color='g',fmt='o',markersize=2)
            grafica.ax.errorbar(self.dosis,self.promsB,yerr=self.erroresB,color='b',fmt='o',markersize=2)
            xasR=np.linspace(self.promsR[0],self.promsR[-1]+0.005,100)
            xasG=np.linspace(self.promsG[0],self.promsG[-1]+0.005,100)
            xasB=np.linspace(self.promsB[0],self.promsB[-1]+0.005,100)
            
        
        yasR=fR(xasR)
        yasG=fG(xasG)
        yasB=fB(xasB)
        

        if self.tipoCanal=="Multicanal":
            SR=np.sum(((self.dosis-fR(self.promsR)))**2)
            SG=np.sum(((self.dosis-fG(self.promsG)))**2)
            SB=np.sum(((self.dosis-fB(self.promsB)))**2)
        else:
            SR=np.sum(((self.dosis-fR(self.netODR)))**2)
            SG=np.sum(((self.dosis-fG(self.netODG)))**2)
            SB=np.sum(((self.dosis-fB(self.netODB)))**2)
            
        bondadR=chi2.sf(SR,len(self.dosis)-len(self.parametrosOptimosR))
        bondadG=chi2.sf(SG,len(self.dosis)-len(self.parametrosOptimosG))
        bondadB=chi2.sf(SB,len(self.dosis)-len(self.parametrosOptimosB))
        
        print(bondadR)
        print(bondadG)
        print(bondadB)
        
        self.labelR=self.labelR+'\n'+r"$r^2=$"+str(bondadR*bondadR)[:5]  
        self.labelG=self.labelG+'\n'+r"$r^2=$"+str(bondadG*bondadG)[:5]  
        self.labelB=self.labelB+'\n'+r"$r^2=$"+str(bondadB*bondadB)[:5]   
            
        grafica.ax.plot(yasR,xasR,'r--',label=self.labelR)
        grafica.ax.plot(yasG,xasG,'g--',label=self.labelG)
        grafica.ax.plot(yasB,xasB,'b--',label=self.labelB)    
            
        grafica.figure.legend(loc=7)
        grafica.figure.tight_layout()
        grafica.figure.subplots_adjust(right=0.75)
        
        
        
        pOptimos=[self.parametrosOptimosR,self.parametrosOptimosG,self.parametrosOptimosB]
        pCova=[self.pCovarianzaR,self.pCovarianzaG,self.pCovarianzaB]
        dopts=[[self.netODR,self.netODG,self.netODB],[self.promsR,self.promsG,self.promsB]]
        ceros=[self.promsRCero,self.promsGCero,self.promsBCero]
        incer=[self.erroresR,self.erroresG,self.erroresB]
        funcionesRGB=[fR,fG,fB]
        labels=[self.labelFuncion,self.labelR,self.labelG,self.labelB]
        guardar_calibracion(self.tipoCanal,
                            self.tipoCurva,
                            pOptimos,
                            pCova,
                            self.dosis,
                            dopts,
                            ceros,
                            incer,
                            funcionesRGB,
                            self.funcionCalculaDelta,
                            self.funcionTaD,
                            labels,
                            nombreArchivo) 

       
        grafica.Show() 
        
                            
                
                
                
                    

            

        
        


