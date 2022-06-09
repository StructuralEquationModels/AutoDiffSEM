############################################################################################
### RAMSymbolicZygote

function objective!(
        semml::SemML,
        par,
        model::AbstractSemSingle,
        has_meanstructure::Val{T},
        imp::RAMSymbolicZygote) where T
    
    let Σ = Σ(imply(model)), Σₒ = obs_cov(observed(model)),
        
        Σ_chol = cholesky(Symmetric(Σ); check = false)

        if !isposdef(Σ_chol) return non_posdef_return(par) end

        ld = logdet(Σ_chol)
        Σ⁻¹ = inv(Σ)

        if T
            let μ = μ(imply(model)), μₒ = obs_mean(observed(model))
                μ₋ = μₒ - μ
                return ld + dot(Σ⁻¹, Σₒ) + dot(μ₋, Σ⁻¹, μ₋)
            end
        else
            return ld + dot(Σ⁻¹, Σₒ)
        end
    end
end