#include<iostream>
#include<vector>
#include<set>
#include <algorithm>
#include "mkl.h"
#include "mkl_cblas.h"
//function to create CSR 
sparse_matrix_t create_mkl_csr(
    int rows,
    int cols,
    const std::vector<MKL_INT>&r ,
    const std::vector<MKL_INT>&c,
    const std::vector<double>&v)
{
    sparse_matrix_t coo,csr;
    mkl_sparse_d_create_coo(&coo,SPARSE_INDEX_BASE_ZERO,rows,cols,r.size(),(MKL_INT*)r.data(),(MKL_INT*)c.data(),(double*)v.data());
    mkl_sparse_convert_csr(coo,SPARSE_OPERATION_NON_TRANSPOSE,&csr);
    mkl_sparse_destroy(coo);
    return csr;
}


//function to solve using Schur method 
void Schur(sparse_matrix_t K,int nx,sparse_matrix_t ALU,std::vector<double>&x)
{
    //size of K
    int n=nx*nx;

    //mid 
    int mid=(nx+1)/2-1;

    //boundaries=union(num(:,mid)',num(mid,:));
    std::set<int> boundary_set;

    for(int i=0;i<nx;i++)
    {
        //add vertical mid line in boundaries_set
        boundary_set.insert(i*nx+mid);

        //add horizontal mid line in boundaries_set
        boundary_set.insert(mid*nx+i);
    }

    std::vector<int> inner_idx,bound_idx;
    std::vector<int>g2i(n,-1); //for inner
    std::vector<int>g2b(n,-1); //for boundaries
    //create sd1,sd2,sd3
    for(int i=0;i<n;i++)
    {
        //check if it belongs to boundary 
        if(boundary_set.count(i))
        {
            //if it belongs to boundary add it to boundary vector
            g2b[i]=bound_idx.size();
            bound_idx.push_back(i);
        }
        else{
            //else it belongs to inner add it to inner vector
            g2i[i]=inner_idx.size();
            inner_idx.push_back(i);
        }    
    }

    //construct A,B,BT,D
    //A=K(inner,inner) , B=K(innder,boundaries), BT=K(boundaries,inner),D=K(boundaries,boundaries)

    //create inner,boundaries sizes for matrices A,B,BT,D
    int nI=inner_idx.size();    
    int nB=bound_idx.size();

    //csr form of A,B,BT,D
    MKL_INT *r_start,*r_end,*cols;
    double* vals;
    sparse_index_base_t indexing;
    MKL_INT rows_k,cols_k;
    mkl_sparse_d_export_csr(K,&indexing,&rows_k,&cols_k,&r_start,&r_end,&cols,&vals);


    //create vector to store rows of A,B,BT,D
    std::vector<MKL_INT> rA,rB,rBT,rD;

    
    //create vector to store cols of A,B,BT,D
    std::vector<MKL_INT> cA,cB,cBT,cD;

    
    //create vector to store values of A,B,BT,D
    std::vector<double> vA,vB,vBT,vD;


    //create the matrices B,BT,D
    for(int i=0;i<n;i++)
    {   
        for(int j=r_start[i];j<r_end[i];j++)
        { 
            int row=i;
            int col=cols[j];
            double value=vals[j];
            //matrix A had dimensions inner,inner
            if(g2i[row]!=-1 && g2i[col]!=-1){rA.push_back(g2i[row]); cA.push_back(g2i[col]); vA.push_back(value);}
            //matrix B has dimensions inner,boundaries
            if(g2i[row]!=-1 && g2b[col]!=-1){rB.push_back(g2i[row]); cB.push_back(g2b[col]); vB.push_back(value);}
            //matrix BT has dimensions boundaries innner
            if(g2b[row]!=-1 && g2i[col]!=-1){rBT.push_back(g2b[row]); cBT.push_back(g2i[col]); vBT.push_back(value);}
            //matrix D has dimensions bounadires,bounadries
            if(g2b[row]!=-1 && g2b[col]!=-1){rD.push_back(g2b[row]); cD.push_back(g2b[col]); vD.push_back(value);}
        }
    }

    //construct the car forms of A,B,BT,D
    sparse_matrix_t A=create_mkl_csr(nI,nI,rA,cA,vA);
    sparse_matrix_t B=create_mkl_csr(nI,nB,rB,cB,vB);
    sparse_matrix_t BT=create_mkl_csr(nB,nI,rBT,cBT,vBT);
    sparse_matrix_t D=create_mkl_csr(nB,nB,rD,cD,vD);

    //construct fI,fB
    std::vector<double>fI(nI);
    std::vector<double>fB(nB);

    //form vector b
    std::vector<double>b(n,0.0);
    std::vector<double>ones(n,1.0); //identity matrix 
    
    
    //descripotr ofr K
    matrix_descr descr_gen={SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_FULL,SPARSE_DIAG_NON_UNIT};

    //b=K*ones
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,K,descr_gen,ones.data(),0.0,b.data());

    //fI=b(inner)
    for(int i=0;i<nI;i++)
    {
        fI[i]+=b[inner_idx[i]];
    }

    //fB=b(boundaries)
    for(int i=0;i<nB;i++)
    {
        fB[i]+=b[bound_idx[i]];
    }

    //fs=fB-BT*(U\(L\fI));
    matrix_descr L={SPARSE_MATRIX_TYPE_TRIANGULAR,SPARSE_FILL_MODE_LOWER,SPARSE_DIAG_NON_UNIT};
    matrix_descr U={SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT};

    std::vector<double> z(nI),tmp(nI),fs(nB);
    //solve L*tmp=fI
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,L,fI.data(),tmp.data());
    //solve U*z=tmp
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,U,tmp.data(),z.data());


    //compute BT*z
    std::vector<double> BTz(nB);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,BT,descr_gen,z.data(),0.0,BTz.data());
    
    //compute fs=fB-BTz
    // for(int i=0;i<nB;i++)
    // {
    //     fs[i]=fB[i]=BTz[i];
    // }

    //fs=fB
    cblas_dcopy(nB,fB.data(),1,fs.data(),1);

    //fs-=BTz
    cblas_daxpy(nB,-1.0,BTz.data(),1,fs.data(),1);


    //compute S matrix 
    std::vector<std::vector<double>> S_dense(nB*nB,0.0);

    /*compute S=D-BT*(U\(L\B))
    S=D-BT*z2
    U*z2=temp2
    L*temp2=B
*/    
    //create vectors z2,temp2,BTZ2,Bj οπου bJ διανυσμα με την j στηλη του Β
    std::vector<double> z2(nI),temp2(nI),BTZ2(nB),Bj(nI,0.0);
    //transfomr D matrix to dense
    for(size_t k=0;k<vD.size(),k++)
    {
        MKL_INT row=rD[k];
        MKL_INT col=cD[k];
        double value=vD[k];

        S_Dense[row+col*nB]=value;
    }


    //solve L*temp2=B
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,L,B.data(),temp2.data());


}
int main()
{


    return 0;
}