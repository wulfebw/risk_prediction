
function get_gmm_dists(mus::Array{Float64}, sigmas::Array{Float64})
    K = size(mus, 2)
    try
        dists = [MvNormal(mus[:,k], sigmas[:,:,k]) for k in 1:K]
    catch e
        println("exception raised while getting the gmm dists: $(e)")
        for i in 1:K
            println("component $(i)")
            println("mus: $(mus[:,i])")
            println("sigmas: $(sigmas[:,:,i])")
        end
        throw(e)
    end
end

function compute_gmm_ll(samples::Array{Float64}, 
        pis::Array{Float64}, 
        mus::Array{Float64}, 
        sigmas::Array{Float64}, 
        samp_w::Array{Float64} = ones(size(samples, 2))
    )
    N, K = size(samples, 2), length(pis)
    dists = get_gmm_dists(mus, sigmas)
    ll = 0
    for sidx in 1:N
        total = 0
        for k in 1:K
            total += pis[k] * pdf(dists[k], samples[:, sidx])
        end
        ll += samp_w[sidx] * log(total)
    end
    return ll
end

function fit_gmm(samples::Array{Float64}; 
        samp_w::Array{Float64} = ones(1, size(samples,2)), 
        max_iters::Int = 30, 
        tol::Float64 = 1e-2, 
        num_components::Int = 2)
    # init
    N, K, D = size(samples, 2), num_components, length(samples[:, 1])
    w = zeros(K, N)
    
    mus = zeros(D, K)
    step = Int(ceil(D / K))
    for k in 1:K
        s = (k-1) * step
        e = s + step
        mus[:,k] = mean(samples[:, s+1:e],2)
    end
    
    sigmas = zeros(D, D, K)
    for k in 1:K
        sigmas[:,:,k] = eye(D)
    end
    pis = rand(K)
    pis ./= sum(pis)
    dists = get_gmm_dists(mus, sigmas)
    prev_ll = compute_gmm_ll(samples, pis, mus, sigmas)
    for iteration in 1:max_iters
        # e-step
        log_pis = log(pis)
        for sidx in 1:N
            for k in 1:K
                w[k, sidx] = log_pis[k] + logpdf(dists[k], samples[:, sidx])
            end
        end
        w = normalize_log_probs(w)
        w .*= samp_w # account for sample probability

        # m-step
        pis = sum(w, 2) ./ sum(w)
        
        mus = zeros(D, K)
        for k in 1:K
            for sidx in 1:N
                mus[:, k] += w[k,sidx] * samples[:,sidx]
            end
            mus[:, k] ./= sum(w[k,:])
        end
        
        sigmas = ones(D,D,K) * 1e-8
        for k in 1:K
            for sidx in 1:N
                diff = samples[:,sidx] - mus[:, k]
                sigmas[:, :, k] += w[k, sidx] * (diff * transpose(diff))
            end
            sigmas[:, :, k] ./= sum(w[k, :])
        end
        
        # check for convergence
        ll = compute_gmm_ll(samples, pis, mus, sigmas, samp_w)
        if abs(ll - prev_ll) < tol
            break
        else
            prev_ll = ll
            dists = get_gmm_dists(mus, sigmas)
        end
    end
    return pis, mus, sigmas
end