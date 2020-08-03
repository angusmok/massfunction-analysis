# massfunction-analysis
This repository contains useful python functions, including some functions for the analysis of mass functions.

List of Functions:

# (1) Math Functions
- simplepowerlaw(M, M0, Gamma)
- truncatedpowerlaw(M, N0, M0, Gamma)
- truncatedpowerlaw_1(M, N0, M0)
- schechter(M, phi, M0, Gamma)
- schechter_1(M, phi, M0)
- schechter_2(M, phi, M0)
- simplepowerlaw_log(logM, M0, Gamma)
- truncatedpowerlaw_log(logM, N0, M0, Gamma)
- truncatedpowerlaw_1_log(logM, N0, M0)
- truncatedpowerlaw_2_log(logM, N0, M0)
- schechter_log(logM, phi, M0, Gamma)
- schechter_1_log(logM, phi, M0)
- schechter_2_log(logM, phi, M0)
- trace_mean(x)
- trace_sd(x)
- linefunction(x, m, b)

# (2) Printing Function
- printarraynumbering(array)

# (3) Declare Simple Functions
- rsquared(x, y)
- reducedchisq(ydata, ymod, dof, sd)
- log10_labels(x, pos)
- find_nearest2(array, value)
- find_nearest2guided(array, value, org_idx)
