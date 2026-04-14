#include<vector>
#include<iostream>
#include<fstream>
#include<mkl.h>
#include<cmath>
#include<cassert>
#include<numbers>
#include<stdexcept>
#include<chrono>

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
std::vector<double> build_rhs(int m,double G , double sigma)
{
    int n=(m-1)*(m-1);

    //L=[-1,1]
    int L1=-1;
    int L2=1;

    //step 
    double h=(L2-L1)/static_cast<double>(m);

    // i need the right side 
    std::vector<double> b(n);

    //coordiantes
    std::vector<double> xx(n) ,yy(n);
    for(int i=0;i<m-1;i++)
    {
        xx[i]=L1+h*(i+1);
        yy[i]=L1+h*(i+1);
    }

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

//CG
std::vector<double> cg_solve(
    sparse_matrix_t A,
    const std::vector<double>& b,
    int n,
    int maxIters,
    double tol)
{
    std::cout << "Starting Conjugate Gradient...\n";

    std::vector<double> x(n, 0.0);      // initial guess
    std::vector<double> r = b;          // r = b - A*x = b
    std::vector<double> p = r;          // p = r
    std::vector<double> Ap(n);

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    double normb = cblas_dnrm2(n, r.data(), 1);
    if (normb == 0.0) normb = 1.0;

    for (int iter = 0; iter < maxIters; iter++)
    {
        double r2 = cblas_ddot(n, r.data(), 1, r.data(), 1);

        // Ap = A * p
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0,
            A,
            descr,
            p.data(),
            0.0,
            Ap.data()
        );

        double pAp = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
        double alpha = r2 / pAp;

        // x = x + alpha*p
        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);

        // r = r - alpha*Ap
        double neg_alpha = -alpha;
        cblas_daxpy(n, neg_alpha, Ap.data(), 1, r.data(), 1);

        double normr = cblas_dnrm2(n, r.data(), 1);
        if (iter % 100 == 0)
            std::cout << " iter " << iter << "  residual = " << normr << "\n";

        if (normr < tol * normb)
        {
            std::cout << "CG converged in " << iter
                      << " iterations, residual = " << normr << "\n";
            return x;
        }

        double r_new = cblas_ddot(n, r.data(), 1, r.data(), 1);
        double beta = r_new / r2;

        // p = r + beta*p
        cblas_daxpby(n, 1.0, r.data(), 1, beta, p.data(), 1);
    }

    std::cout << "CG did NOT converge within " << maxIters << " iterations\n";
    return x;
}

//main function 
int main()
{
    auto start=std::chrono::high_resolution_clock::now();
    std::cout<<"simulation starting\n";
    //dimensions
    int m=200;
    int n=(m-1)*(m-1);

    //variables for function f 
    double G=1.0,sigma=0.1;

    double L1=-1,L2=1;
    double h=(L2-L1)/m;
    
    std::vector<double> xx(m-1),yy(m-1);
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
    


    sparse_index_base_t indexing ;
    MKL_INT rows_csr, cols_csr;
    MKL_INT *row_ptr, *row_end_unused, *col_ind;
    double *val_csr;


 mkl_sparse_d_export_csr(Ahandle, &indexing,
        &rows_csr, &cols_csr,
        &row_ptr, &row_end_unused,
        &col_ind, &val_csr);


    //build rhs 
    std::vector<double>b=build_rhs(m,G,sigma);


    //solve 
   std::vector<double> x =cg_solve(Ahandle,b, n, 50000, 1e-5);

    mkl_sparse_destroy(Ahandle);

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
