# Portfolio AAD & Universal Risk Engine (`spxa.risk`)

The `spxa.risk` module delivers Tier-1 institutional risk analytics, real-time XVA modeling, regulatory margin and capital engines (ISDA-SIMM and FRTB), and automated Adjoint Algorithmic Differentiation (AAD) across complex multi-asset derivative portfolios.

## Architectural Capabilities

### 1. Financial Instruments & Bilateral Portfolios (`spxa.risk.instruments`, `spxa.risk.portfolio`)
- **Derivatives**: `EuropeanOption`, `BarrierOption` (discrete/continuous monitoring with in-out parity), `AsianOption` (running arithmetic average), `BasketOption`, `InterestRateSwap` (IRS), and `CreditDefaultSwap` (CDS).
- **Netting Sets & CSA**: Bilateral netting agreements governing independent amounts, uncollateralized thresholds ($H_C$), Minimum Transfer Amounts (MTA), and Margin Periods of Risk (MPOR).
- **Dynamic Exposure Profiles**: Time-dependent Expected Exposure ($\text{EE}(t)$), Potential Future Exposure ($\text{PFE}_{95}(t)$, $\text{PFE}_{99}(t)$), and Expected Positive Exposure ($\text{EPE}$).

### 2. Portfolio Adjoint Algorithmic Differentiation (AAD) (`spxa.risk.aad`)
Computes the full portfolio Present Value, Delta vector, Vega matrix, and second-order Cross-Gamma Hessian matrix in a single reverse-mode backward pass:
$$\boldsymbol{\Delta} = \nabla_{\boldsymbol{S}_0} \Pi_0, \quad \boldsymbol{\mathcal{V}} = \nabla_{\boldsymbol{\sigma}} \Pi_0, \quad \boldsymbol{\Gamma} = \nabla^2_{\boldsymbol{S}_0} \Pi_0$$

Includes smooth mollifiers (`smooth_indicator`) for robust differentiation through non-smooth barrier and digital contract boundaries.

### 3. Real-Time XVA Engine (`spxa.risk.xva`)
Evaluates comprehensive bilateral valuation adjustments:
$$\text{Total XVA} = \text{CVA} - \text{DVA} + \text{FVA} + \text{MVA} + \text{KVA}$$
- **Credit Valuation Adjustment (CVA)**: Counterparty default risk under collateralization.
- **Debit Valuation Adjustment (DVA)**: Own-default risk and credit benefit.
- **Funding Valuation Adjustment (FVA)**: Funding Cost Adjustment (FCA) and Funding Benefit Adjustment (FBA) over uncollateralized exposure.
- **Margin Valuation Adjustment (MVA)**: Funding cost of dynamic initial margin posting over the trade lifecycle.
- **Capital Valuation Adjustment (KVA)**: Hurdle cost of holding economic and regulatory capital.

### 4. Regulatory Margin & Capital (`spxa.risk.frtb_simm`)
- **ISDA SIMM v2.6+ Engine**: Multi-class margin aggregation (Equity, Interest Rate, Credit, Commodity, FX) incorporating intra-bucket and cross-class correlation matrices.
- **FRTB Standardized Approach (SA)**: Sensitivities-Based Method (SBM) capital requirements across Low, Medium, and High correlation scenarios with Default Risk Charges (DRC).

### 5. Universal Scenario Generator & Stress Testing (`spxa.risk.scenarios`, `spxa.risk.reporting`)
- Historical crisis replay (`LEHMAN_2008`, `COVID_MARCH_2020`, `INFLATION_SHOCK_2022`).
- Reverse stress testing identifying minimal market displacements that breach loss thresholds.
- Executive risk reporting (`format_risk_executive_summary`).
