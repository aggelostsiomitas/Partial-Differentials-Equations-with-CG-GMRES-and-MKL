#include<vector>
#include<mkl.h>
#include<iostream>
#include<limits>
#include<iomanip>
#include<cmath>


void  ARNOLDI(const std::vector<double>&A , const std::vector<double>& r, int& m,std::vector<double>&V, std::vector<double>&H)
{
    //size of r 
    int n=r.size();

    //H=zeros(m+1,m)
    H.assign((m+1)*m,0.0);

    //V=zeros(n,m+1)
    V.assign((m+1)*n,0.0);

    //V(:,1)=r/normr
    //i will copy the r to the first row of V V(:,1)
    cblas_dcopy(n,r.data(),1,V.data(),1);
    
    //calculate norm of r 
    double normr=cblas_dnrm2(n,V.data(),1);

    //divide the r with normr
    cblas_dscal(n,1.0/normr,V.data(),1);


    //iterate 
    for(int j=0;j<m;j++)
    {
        //get product V(:,j+1)=A*V(:,j)
        //vector=A*vector 

        //V(:,j)
        double* v_current=&V[j*n];

        //V(:,j+1)
        double* v_next=&V[(j+1)*n];

        //v_next=A*v_current
        cblas_dgemv(CblasColMajor,CblasNoTrans,n,n,1.0,A.data(),n,v_current,1,0.0,v_next,1);

        for(int i=0;i<=j;i++)
        {
            //V(:,i)
            double* v_i=&V[i*n];

            //H(i,j)=V(:,j+1)*V(:,i)
            //H(i,j)=v_next*v_i
            //column major approach
            double hij=cblas_ddot(n,v_next,1,v_i,1);
            H[i+j*(m+1)]=hij;

            //V(:,j+1)=V(:,j+!)-H(i,j)*V(:,i)
            //v_next=v_next-hij*v_i
            cblas_daxpy(n,-hij,v_i,1,v_next,1);
        }


        //get norm of vector 
        //H(j+1,j)=norm(V(:,j+1));
        double normV=cblas_dnrm2(n,v_next,1);
        H[(j+1)+j*(m+1)]=normV;

        if(std::abs(normV)<std::sqrt(std::numeric_limits<double>::epsilon()))
        {
            m=j;
            H.resize(m*(m-1));
            V.resize(n*m);
            return;
        }

        //V(:,j+1)=V(:,j+1)/H(j+1,j)
        cblas_dscal(n,1.0/normV,v_next,2);
    }
}