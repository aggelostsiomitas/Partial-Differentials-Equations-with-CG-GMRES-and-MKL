#include<vector>
#include<iostream>
#include<fstream>
#include<cmath>
#include<cassert>
#include<numbers>
#include<stdexcept>
#include<chrono>
#include<mkl.h>
#include "mkl_spblas.h"

//built laplacian matrix in COO format 
void build_laplacian_coo(
    int m,
    std::vector<MKL_INT>& rows,
    std::vector<MKL_INT>& cols,
    std::vector<double>& vals)
{
    const int W = m - 1;      
    const int n = W * W;

    rows.clear();
    cols.clear();
    vals.clear();
    rows.reserve(n * 5);
    cols.reserve(n * 5);
    vals.reserve(n * 5);

    for (int r = 0; r < W; ++r)
    {
        for (int c = 0; c < W; ++c)
        {
            const int k = r * W + c;

            // center
            rows.push_back(k);
            cols.push_back(k);
            vals.push_back(-4.0);

            // left neighbor
            if (c > 0) {
                const int j = k - 1;
                rows.push_back(k); cols.push_back(j); vals.push_back(1.0);
            }

            // right neighbor
            if (c + 1 < W) {
                const int j = k + 1;
                rows.push_back(k); cols.push_back(j); vals.push_back(1.0);
            }

            // up neighbor
            if (r > 0) {
                const int j = k - W;
                rows.push_back(k); cols.push_back(j); vals.push_back(1.0);
            }

            // down neighbor
            if (r + 1 < W) {
                const int j = k + W;
                rows.push_back(k); cols.push_back(j); vals.push_back(1.0);
            }
        }
    }
}




//build rhs 
std::vector<double> build_rhs(int m,double G , double sigma,const std::vector<double>&xx,const std::vector<double>&yy)
{
    int n=(m-1)*(m-1);

    //L=[-1,1]
    int L1=-1;
    int L2=1;

    //step 
    double h=(L2-L1)/static_cast<double>(m);

    // i need the right side 
    std::vector<double> b(n);

 

    //make the f 
    auto f=[&](double X,double Y)
    {
        return 4*std::numbers::pi*G*(1/std::sqrt(std::numbers::pi*sigma))*std::exp(-(X*X+Y*Y)/(2*sigma*sigma));
    };

    //create the f(x,y)
    for(int r=0;r<m-1;r++)
    {
        for(int c=0;c<m-1;c++)
        {
            int k=r*(m-1)+c;
            b[k]=h*h*f(xx[c],yy[r]);
        }
    }
    return b;
}


//partition function for quicksort
int partition(std::vector<int>&arr1 ,std::vector<double>&arr2 ,int low,int high)
{
    //choose high as the pivot 
    //arr1=column indices vector
    int pivot=arr1[high];

    //bigger value is none
    int i=low-1;

    //iterate column for sort 
    for(int j=low;j<=high-1;j++)
    {
        if(arr1[j]<pivot)
        {
            i++;

            //if i found a column with higher value swap
            int tempCol=arr1[i];
            arr1[i]=arr1[j];
            arr1[j]=tempCol;

            //swap the vaule 
            double tempVal=arr2[i];
            arr2[i]=arr2[j];
            arr2[j]=tempVal;
         }
    }

    //swap pivot 
    int tempCol=arr1[i+1];
    arr1[i+1]=arr1[high];
    arr1[high]=tempCol;

    double  tempVal=arr2[i+1];
    arr2[i+1]=arr2[high];
    arr2[high]=tempVal;

    return i+1;
}


//quicksort
void qsort(std::vector<int>&arr1 ,std::vector<double>&arr2,int low,int high)
{
    if(low<high)
    {
        //run partition function
        int pivot=partition(arr1,arr2,low,high);

        //run quick sort 
        qsort(arr1,arr2,low,pivot-1);
        qsort(arr1,arr2,pivot+1,high);

    }
}


//function that will sort every row using qsort
void sort_all_rows( 
    std::vector<int> &col,
    std::vector<double>&values,
    std::vector<int>&row_ptr,
    int n)
{
    for(int r=0;r<n;r++)
    {
        int start=row_ptr[r];
        int end=row_ptr[r+1]-1;

        if(end>start)
        {
            qsort(col,values,start,end);
        }
    }
}

//CG Preconditioned
std::vector<double> cg_precon(
    sparse_matrix_t A,
    sparse_matrix_t ALU,
    const std::vector<double>&b,
    double tol,
    int maxit)

{
    //length of A
    int n=b.size();

    std::cout<<"Starting the preconditioned CG\n";

    
    std::vector<double>x(n,0.0); //initial guess
    std::vector<double> r=b; //residual
    std::vector<double> Ap(n,0.0); //vector that will store the A*x
    std::vector<double> z(n,0.0); //vector that will store the z
    std::vector<double> p(n,0.0); //vector that will store the p
    std::vector<double> temp(n, 0.0); //vector that will store the temp
    // for the  z=U\(L\r) ->  z=U\temp , temp=L\r


    //descriptors 
    matrix_descr descrA,descrL,descrU;

    //descriptor for A
    descrA.type=SPARSE_MATRIX_TYPE_GENERAL;

    //descriptor for L
    descrL.type=SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrL.mode=SPARSE_FILL_MODE_LOWER;
    descrL.diag=SPARSE_DIAG_UNIT;
    //descriptor for U
    descrU.type=SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrU.mode=SPARSE_FILL_MODE_UPPER;
    descrU.diag=SPARSE_DIAG_NON_UNIT;

    //r=b-Ax 
    //initial guess x=0 so r=b-Ax=b
    //norm of b
    double normb=cblas_dnrm2(n,r.data(),1);

    //compute z=U\(L\r)
    //compute temp=L\r
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,descrL,r.data(),temp.data());

    //compute z=U\temp
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,descrU,temp.data(),z.data());

    //p=z
    p=z;

    //iterate 
    for(int i=0;i<maxit;i++)
    {
        //r2=r'*r
        double r2=cblas_ddot(n,r.data(),1,z.data(),1);

        //compute t=A*p
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descrA,p.data(),0.0,Ap.data());

        //a=r2/(p'*t)
        //compute t1=p'*t 
        double t1=cblas_ddot(n,p.data(),1,Ap.data(),1);

        //compute a
        double alpha=r2/t1;

        //update solution x=x+a*p
        cblas_daxpy(n,alpha,p.data(),1,x.data(),1);

        //update residual r=r-at
        cblas_daxpy(n,-alpha,Ap.data(),1,r.data(),1);

        //precondition again z=U\(L\r)
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,descrL,r.data(),temp.data());
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,ALU,descrU,temp.data(),z.data());

        //norm of r 
        double normr=cblas_dnrm2(n,r.data(),1);

        if (i % 100 == 0) {
            std::cout << "Iteration " << i << ", Relative Residual: " << normr / normb << "\n";
        }
        //check convergence 
        if(normr<tol*normb)
        {
            std::cout << "PCG converged in " << i << " iterations." << std::endl;
            std::cout<<"Converged after :"<<i<<" iterations\n";
            std::cout<<"Residual: "<<normr<<'\n';
            return x;
        }

        //projection coeeficient 
        //b=(r'*z/r2);
        
        //compute r'*z
        double rz=cblas_ddot(n,r.data(),1,z.data(),1);
        double beta=rz/r2;

        //new direction p=z+b*p
        cblas_daxpby(n,1.0,z.data(),1,beta,p.data(),1);
    }
    return x;
};


//main function 
int main()
{
    auto start=std::chrono::high_resolution_clock::now();
    std::cout<<"simulation starting\n";
    //dimensions
    int m=200;
    int n=(m-1)*(m-1);
    std::cout<<"starting simulation with n= "<<n<<'\n';

    //variables for function f 
    double G=1.0,sigma=0.1;

    double L1=-1,L2=1;
    double h=(L2-L1)/m;
    
       //coordiantes
    std::vector<double> xx(m-1) ,yy(m-1);
    for(int i=0;i<m-1;i++)
    {
        xx[i]=L1+h*(i+1);
        yy[i]=L1+h*(i+1);
    }

    //build the laplacian in coo
    std::vector<MKL_INT>rows,cols;
    std::vector<double>vals;
    build_laplacian_coo(m,rows,cols,vals);
 

    //create MKL COO matrix 
    sparse_matrix_t Acoo;
    mkl_sparse_d_create_coo(&Acoo,SPARSE_INDEX_BASE_ZERO,n,n,vals.size(),rows.data(),cols.data(),vals.data());

    //convert COO to CSR
    sparse_matrix_t Ahandle ;
    mkl_sparse_convert_csr(Acoo,SPARSE_OPERATION_NON_TRANSPOSE,&Ahandle);
    mkl_sparse_destroy(Acoo);
    

    //build rhs 
    std::vector<double>b=build_rhs(m,G,sigma,xx,yy);


     //solve 
   //std::vector<double> x =cg_solve(Ahandle,b, n, 50000, 1e-5);


    sparse_index_base_t indexing;
    MKL_INT rows_csr, cols_csr;
    MKL_INT *row_ptr, *row_end, *col_ind;
    double *val_csr;

  mkl_sparse_d_export_csr(Ahandle, &indexing, &rows_csr, &cols_csr,
                        &row_ptr, &row_end, &col_ind, &val_csr);

// nnz = total nonzeros
MKL_INT nnz = row_end[rows_csr - 1];

// φτιάχνω ΚΑΝΟΝΙΚΟ CSR row_ptr (μέγεθος rows_csr+1)
std::vector<int> row_ptr_vec(rows_csr + 1);

// 0-based περίπτωση (που έχεις εδώ)
if (indexing == SPARSE_INDEX_BASE_ZERO) {
    row_ptr_vec[0] = 0;
    for (int i = 0; i < rows_csr; ++i) {
        row_ptr_vec[i+1] = row_end[i];   // row_end[i] = cumulative nnz μέχρι τη γραμμή i
    }
} else {
    // 1-based περίπτωση (αν ποτέ προκύψει)
    row_ptr_vec[0] = 1;
    for (int i = 0; i < rows_csr; ++i) {
        row_ptr_vec[i+1] = row_end[i];
    }
}

// τώρα αντιγράφω col/val (0-based ακόμα)
std::vector<int>    col_ind_vec(col_ind, col_ind + nnz);
std::vector<double> val_vec    (val_csr, val_csr + nnz);

// sort ανά γραμμή σε 0-based CSR
sort_all_rows(col_ind_vec, val_vec, row_ptr_vec, rows_csr);

//μετατροπή σε 1-based για ILU0
if (indexing == SPARSE_INDEX_BASE_ZERO) {
    for (int i = 0; i <= rows_csr; ++i)
        row_ptr_vec[i]++;      

    for (int i = 0; i < nnz; ++i)
        col_ind_vec[i]++;     
}

// ILU0 πάνω σε 1-based, sorted CSR
std::vector<double> bilu0 = val_vec;

MKL_INT ipar[128];
double  dpar[128];
for (int i = 0; i < 128; ++i) { ipar[i] = 0; dpar[i] = 0.0; }

MKL_INT ierr;

dcsrilu0(&rows_csr,
         bilu0.data(),
         row_ptr_vec.data(),
         col_ind_vec.data(),
         bilu0.data(),
         ipar, dpar, &ierr);

if (ierr != 0) {
    std::cout << "ILU0 failed with error code: " << ierr << "\n";
} else {
    std::cout << "ILU0 factorization successful!\n";
}

// ALU από ILU0
sparse_matrix_t ALU;
mkl_sparse_d_create_csr(
    &ALU,
    SPARSE_INDEX_BASE_ONE,
    rows_csr,
    cols_csr,
    row_ptr_vec.data(),
    row_ptr_vec.data() + 1,
    col_ind_vec.data(),
    bilu0.data()
);



    int krylov_dim = 20;
    int max_iters = 1000;
    double tolerance = 1e-5;
    std::vector<double> x = cg_precon(Ahandle, ALU, b, tolerance, max_iters);


    mkl_sparse_destroy(Ahandle);
    mkl_sparse_destroy(ALU);


   std::ofstream out("solution.dat");
   if(!out)
   {
    throw std::runtime_error("failed to open the file\n");
   }

   out.precision(10);
   for(int r=0;r<m-1;r++)
   {
    for(int c=0;c<m-1;c++)
    {
        int k=r*(m-1)+c;
        out<<xx[c]<<" "<<yy[r]<<" "<<x[k]<<"\n";
    }
   }
   out.close();

   std::cout<<"solutions written in solution.dat\n";
    
   auto end=std::chrono::high_resolution_clock::now();
   std::chrono::duration<double>elapsed=end-start;
   std::cout<<"time needed for the simulation: "<<elapsed.count()<<'\n';
    return 0;   
}
