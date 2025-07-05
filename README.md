List of Functions:

# Main Functions
- rungalaxy_output(galname, galnameout, np_SC_C_array, SC_C_complimits, flag_massrange = 1, flag_addbin = 0, flag_makesubplots = 0)

# Sub-functions (plotaxes_agemass)
- plotaxes_agemass(gal_array, galname, complimits, plotaxes, flag, clustermarker = 'ro', plotlegend = True)

# Sub-functions (plotaxes_mspecfit)
- plotaxes_mspecfit(gal_array, galname, complimits, plotaxes, ageflag, errorflag, sigma, flag_mspecfitplot = 1)
- plotgal_likelihood(gal_array, galname, complimits, plotrow, axes, flag)

# Sub-functions (plotaxes_likelihood)
- plotaxes_likelihood(gal_array, galname, complimits, plotaxes, ageflag, loc, sigma, output, outputfile, cutoff_mass_mult = 100, flag_simple = 0, age_flag = 1)
- likelihoodplot(array, galname, agename, plotaxes, loc, sigma, output_flag, outputfile, cutoff_mass_mult = 100, flag_simple = 0, age_flag = 1)

# Sub-functions (plotaxes_histogram/equalhistogram)
- plotaxes_histogram(gal_array, galname, complimits, plotaxes, ageflag, outflag)
- plotaxes_equalhistogram(gal_array, galname, complimits, plotaxes, ageflag, outflag, numgal_bin_in = 5, outputbinstofile = False)
- histogramplot(galname, age_label, bins, n_dM, bins_fit, n_fit_dM, n_fit_dM_err, complimits, array_plot, plotaxes, len_sortedarray, outflag)
- makearrayhist(array1, array2, mass_lim, massindex = 4)
- makearrayhistequal(array1, mass_lim, flag, numgal_bin_in = 5, massindex = 4)
- curve_fit3(array1, array2, array3, mass_lim, flag, flag_plot, array2_err)

# Other Required Functions
- agecuts_outputarrays(gal_array, galname, complimits, ageflag)
- is_float(element: any)
- converttonumpy(array, arrayname, flag_output = True)
- converttonumpy_float(array, arrayname, flag_output = False)
- log10_labels(x, pos)
- ageflagconvert(agename)
- find_nearest(array, value)
- find_nearest2guided(array, value, org_idx)
- galnameoutfun(galname)

# Other Math Functions
- reducedchisq(ydata, ymod, dof, sd)
- simplepowerlaw(M, M0, Gamma)
- truncatedpowerlaw(M, N0, M0, Gamma)
- schechter(M, phi, M0, Gamma)
- simplepowerlaw_log(logM, M0, Gamma)
- truncatedpowerlaw_log(logM, N0, M0, Gamma)
- schechter_log(logM, phi, M0, Gamma)