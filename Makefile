# The basic common commands to compile
COMMON_COMMAND=-rdc=true -lcudadevrt -lcuda -lcudart -lcufft -lnvrtc --use_fast_math -O4 -ldl
compiler=nvcc
#Set common commands according to the version of cuda
cuda_version=$(shell $(compiler) -V | grep -oE "release [0-9]+.[0-9]+" | grep -oE "[0-9]+.[0-9]+")

ifeq ($(shell echo "$(cuda_version) >= 11.0" | bc), 1)
COMMON_COMMAND+=-std=c++14
ifndef CUDA_ARCH_BIN
COMMON_COMMAND+=-arch=sm_52 -DCUDA_ARCH_BIN=52
endif
else
COMMON_COMMAND+=-std=c++11
ifndef CUDA_ARCH_BIN
COMMON_COMMAND+=-arch=sm_50 -DCUDA_ARCH_BIN=50
endif
endif
ifdef CUDA_ARCH_BIN
COMMON_COMMAND+=-arch=sm_$(CUDA_ARCH_BIN) -DCUDA_ARCH_BIN=$(CUDA_ARCH_BIN)
endif
ifdef USE_FGM
COMMON_COMMAND+=-lcublas -lcusparse
endif

all: SPONGE SPONGE_TI
	

install: all
	

MD_OBJECTS = main.o common.o control.o MD_core/MD_core.o bond/bond.o bond/bond_soft.o bond/listed_forces.o angle/angle.o angle/Urey_Bradley_force.o dihedral/dihedral.o dihedral/improper_dihedral.o cmap/cmap.o nb14/nb14.o neighbor_list/neighbor_list.o No_PBC/Coulomb_Force_No_PBC.o No_PBC/generalized_Born.o No_PBC/Lennard_Jones_force_No_PBC.o Lennard_Jones_force/Lennard_Jones_force.o Lennard_Jones_force/solvent_LJ.o Lennard_Jones_force/LJ_soft_core.o Lennard_Jones_force/pairwise_force.o PME_force/PME_force.o  thermostats/Langevin_MD.o thermostats/Middle_Langevin_MD.o thermostats/Andersen_thermostat.o thermostats/Berendsen_thermostat.o thermostats/nose_hoover_chain.o barostats/MC_barostat.o barostats/Berendsen_barostat.o barostats/andersen_barostat.o restrain/restrain.o constrain/constrain.o constrain/simple_constrain.o constrain/SETTLE.o constrain/SHAKE.o virtual_atoms/virtual_atoms.o SITS/SITS.o collective_variable/collective_variable.o collective_variable/simple_cv.o collective_variable/RMSD.o collective_variable/tabulated.o collective_variable/combine.o bias/restrain_cv.o bias/steer.o bias/Meta1D.o wall/hard_wall.o wall/soft_wall.o plugin/plugin.o
ifdef USE_FGM
MD_OBJECTS += FGM_surface.o
endif
-include $(MD_OBJECTS:.o=.d)

SPONGE: $(MD_OBJECTS)
	sed -i 's/\r$///' mdin.txt
	sed -i 's/\r$///' covid-tip4p/*
	$(compiler) -o $@  $^ $(COMMON_COMMAND)
		

TI_OBJECTS = main_ti.o common.o control.o MD_core/MD_core.o neighbor_list/neighbor_list.o  Lennard_Jones_force/LJ_soft_core.o PME_force/PME_force.o PME_force/cross_PME.o
-include $(TI_OBJECTS:.o=.d)

SPONGE_TI: $(TI_OBJECTS)
	$(compiler) -o $@  $^ $(COMMON_COMMAND)

SUBDIRS=$(shell ls -l | grep ^d | awk '{print $$9}')
clean:
	rm -f *.d
	rm -f *.d.*
	rm -f *.o
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.o)
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.d)
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.d.*)

-include $(foreach i, $(SUBDIRS), $(i)/*.Makefile)

%.d: %.cu
	$(info analyzing dependency of $< to $@)
ifeq ($(shell echo "$(cuda_version) > 10.0" | bc), 1)
	@set -e; rm -f $@; \
	$(compiler) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
else
	@set -e; rm -f $@; \
	echo $(subst .d,.o,$@) $@: $(subst .d,.cu,$@) $(subst .d,.cuh,$@) control.cuh common.cuh > $@
endif

%.o: %.cu
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)
