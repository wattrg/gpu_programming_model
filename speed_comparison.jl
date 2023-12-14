using Plots
using LaTeXStrings

categories_matrix = ["Julia kernel", "kokkos kernel", "Julia built-in", "Kokkos built-in"]
speeds_matrix = [183.308, 103.894, 69.653, 81.08]

categories_flux = ["Julia", "Kokkos"]
speeds_flux = [5.265, 0.509]

matrix = bar(categories_matrix, speeds_matrix, legend=false, ylabel=L"Execution time ($\mu s$)", dpi=600)
flux = bar(categories_flux, speeds_flux, legend=false, ylabel=L"Execution time ($ms$)", dpi=600)

savefig(matrix, "matrix_speeds.png")
savefig(flux, "flux_speeds.png")
