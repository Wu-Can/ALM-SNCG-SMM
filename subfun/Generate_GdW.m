%%=========================================================================
%% Generate_GdW:
%% Compute the value of the linear operator G(dW), 
%% G:R^{p*q} -> R^{p*q} is an element of the Clarke generalized 
%% Jacobian of the metric projection onto the spectral norm ball 
%% at some matrix X in R^{p*q}.
%% dW = a p-by-q matrix 
%% 
%% GdW = Generate_GdW(dW,options,parPX)
%%
%% Input:
%% dW = a p-by-q matrix
%% options = a structure containing all subparts and index sets 
%%           needed by G(dW)
%% parPX.U = left singular vectors of X
%% parPX.V = right singular vectors of X
%% Output:
%% GdW = GdW_part1 + GdW_part2
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% See Subsubsection 3.1.1 of the paper for more details: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function GdW = Generate_GdW(dW,options,parPX)
p = options.p;
q = options.q;
flag_k1k2 = options.flag_k1k2;

XI1_alpbet2 = options.XI1_alpbet2;
XI2_alpalp = options.XI2_alpalp;
XI2_alpbet1 = options.XI2_alpbet1;
XI2_alpbet2 = options.XI2_alpbet2;
index_alp = options.index_alp;
index_bet1 = options.index_bet1;
index_bet2 = options.index_bet2;
k1 = options.k1;
U = parPX.U;
V = parPX.V;
U_alp = options.U_alp; U_bet2 = options.U_bet2; 
V_alp = options.V_alp; V_bet2 = options.V_bet2; 
U_bet1 = options.U_bet1; V_bet1 = options.V_bet1;
U_alpDiagd = options.U_alpDiagd;
DiagdV_alpT = options.DiagdV_alpT;
Method_flag = options.Method_flag;

%% Compute GdW
if Method_flag == 1
    U_alpTH = U_alp'*dW;
    HV_alp = dW*V_alp;
    U_alpTHV = U_alpTH*V;
    H1_alpalp = U_alpTHV(:,index_alp);
    H1_bet2alp = U_bet2'*HV_alp;
    
    if (flag_k1k2 == 1)
        H1_alpbet1 = U_alpTHV(:,index_bet1);
        H1_bet1alp = U_bet1'*HV_alp;
        
        SH1_alpbet1 = (H1_alpbet1 + H1_bet1alp')/2;
        SH1_bet1alp = SH1_alpbet1';
        TH1_alpbet1 = (H1_alpbet1 - H1_bet1alp')/2;
        TH1_bet1alp = (H1_bet1alp - H1_alpbet1')/2;
    else
        H1_bet1alp = [];
    end
    
    % Generate the parts of H1
    % H1 = U'*dW*V; H1T = H1';
    % TH1 = (H1 - H1T)/2; SH1 = (H1 + H1T)/2;

    H1_alpbet2 = U_alpTHV(:,index_bet2);
    SH1_alpalp = (H1_alpalp + H1_alpalp')/2;
    SH1_alpbet2 = (H1_alpbet2 + H1_bet2alp')/2;
    SH1_bet2alp = SH1_alpbet2';
    
    TH1_alpalp = (H1_alpalp - H1_alpalp')/2;
    TH1_alpbet2 = (H1_alpbet2 - H1_bet2alp')/2;
    TH1_bet2alp = (H1_bet2alp - H1_alpbet2')/2;
    
    % Compute GdW_part2
    if p == q
        GdW_part2 = 0;
    elseif p < q
        GdW_part2 = U_alpDiagd*(U_alpTH - U_alpTHV*V'); 
    else 
        GdW_part2 = (HV_alp - U*[H1_alpalp; H1_bet1alp; H1_bet2alp])*DiagdV_alpT;
    end
    
    % Compute GdW_part1     
    if k1 == 0
        GdW = dW;
    else
        if (flag_k1k2 == 0)
            GdW_part1 = (U_alp*(SH1_alpalp + XI2_alpalp.*TH1_alpalp)+U_bet2*(XI1_alpbet2'.*SH1_bet2alp + XI2_alpbet2'.*TH1_bet2alp))*V_alp';
            GdW_part1 = GdW_part1 + U_alp*((XI1_alpbet2.*SH1_alpbet2 + XI2_alpbet2.*TH1_alpbet2)*V_bet2');
        else
            GdW_part1 = (U_alp*(SH1_alpalp + XI2_alpalp.*TH1_alpalp)+U_bet1*(SH1_bet1alp + XI2_alpbet1'.*TH1_bet1alp)+...
                U_bet2*(XI1_alpbet2'.*SH1_bet2alp + XI2_alpbet2'.*TH1_bet2alp))*V_alp';
            GdW_part1 = GdW_part1 + (U_alp*(SH1_alpbet1 + XI2_alpbet1.*TH1_alpbet1))*V_bet1' + (U_alp*(XI1_alpbet2.*SH1_alpbet2 + XI2_alpbet2.*TH1_alpbet2))*V_bet2';
        end
        GdW = dW - GdW_part1 - GdW_part2;
    end
else                                                                         % Method_flag = 2
    % Generate the parts of H1
    if p <= q
        UTdW = U'*dW; H1 = UTdW*V;
    else
        dWV = dW*V; H1 = U'*dWV;
    end
    SH1 = (H1 + H1')/2;
    TH1 = (H1 - H1')/2;
    H1_bet2bet2 = H1(index_bet2,index_bet2);
    if flag_k1k2 == 1
        H1_bet1bet1 = H1(index_bet1,index_bet1);
        H1_bet2bet1 = H1(index_bet2,index_bet1);
        H1_bet1bet2 = H1(index_bet1,index_bet2);
    end
    
    SH1_alpbet2 = SH1(index_alp, index_bet2);
    SH1_bet2alp = SH1_alpbet2';
    TH1_alpalp = TH1(index_alp,index_alp);
    TH1_alpbet1 = TH1(index_alp, index_bet1);
    TH1_alpbet2 = TH1(index_alp, index_bet2);
    TH1_bet1alp = TH1(index_bet1,index_alp);
    TH1_bet2alp = TH1(index_bet2,index_alp);
    
    % Compute GdW_part2
    if p == q
        GdW_part2 = 0;
    elseif p < q
        GdW_part2 = [U_alpDiagd, U_bet1, U_bet2]*(UTdW - H1*V');
    else 
        GdW_part2 = (dWV - U*H1)*[DiagdV_alpT; V_bet1'; V_bet2'];
    end
    
    % Compute GdW_part1
    if k1 == 0
        GdW = dW;
    else
        if (flag_k1k2 == 0)
            GdW_part1 = (U_alp*(XI2_alpalp.*TH1_alpalp) + U_bet2*(XI1_alpbet2'.*SH1_bet2alp+XI2_alpbet2'.*TH1_bet2alp))*V_alp';
            GdW_part1 = GdW_part1 + (U_alp*(XI1_alpbet2.*SH1_alpbet2+XI2_alpbet2.*TH1_alpbet2) + U_bet2*H1_bet2bet2)*V_bet2';
        else
            GdW_part1 = (U_alp*(XI2_alpalp.*TH1_alpalp) + U_bet1*(XI2_alpbet1'.*TH1_bet1alp)...
                + U_bet2*(XI1_alpbet2'.*SH1_bet2alp+XI2_alpbet2'.*TH1_bet2alp))*V_alp';
            GdW_part1 = GdW_part1 + (U_alp*(XI2_alpbet1.*TH1_alpbet1) + U_bet1*H1_bet1bet1 + U_bet2*H1_bet2bet1)*V_bet1'+...
                (U_alp*(XI1_alpbet2.*SH1_alpbet2+XI2_alpbet2.*TH1_alpbet2) + U_bet1*H1_bet1bet2 + U_bet2*H1_bet2bet2)*V_bet2';
        end
        GdW = GdW_part1 + GdW_part2;
    end
   
end










