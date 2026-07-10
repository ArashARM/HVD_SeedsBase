import torch

from neuraltomo_fem.FE import FE


class NeuralTOMOFEM:
    """
    Minimal callable FEM:
      stress, compliance = fem(density, fiber_dir)
    """
    def __init__(self, problem, device="cpu", isotropic=False):
        """
        `problem` is whatever FE(...) expects in their code.
        In NeuralTOMO this comes from their settings factory.
        If you don’t want their settings system, we can build a tiny Problem class later.
        """
        self.device = torch.device(device)
        self.fe = FE(problem, device=str(self.device))

        self.isotropic = isotropic

    def __call__(self, density: torch.Tensor, phi: torch.Tensor, theta: torch.Tensor, penal=3):
        density = density.to(self.device).flatten().float()
        phi = phi.to(self.device).flatten().float()
        theta = theta.to(self.device).flatten().float()
        # exact call used in NeuralTOMO:
        stress, compliance = self.fe.solve_stress_new(
            phi, theta, density, penal=penal, isotropic=self.isotropic
        )
        return stress, compliance
    



            # self.fem.mesh.mesh: dict containing mesh definition (e.g. {'nelx':60, 'nely':20, 'nelz':10, 'elemSize':[1.0,1.0,1.0]})
        # self.fem.mesh.nelx: number of finite elements in x-direction (e.g. 60)
        # self.fem.mesh.nely: number of finite elements in y-direction (e.g. 20)
        # self.fem.mesh.nelz: number of finite elements in z-direction (e.g. 10)
        # self.fem.mesh.elemSize: physical size of each element [dx, dy, dz] (e.g. [1.0, 1.0, 1.0] meters)
        # self.fem.mesh.numElems: total number of elements in the mesh = nelx*nely*nelz (e.g. 60*20*10 = 12000)
        # self.fem.mesh.numNodes: total number of nodes = (nelx+1)*(nely+1)*(nelz+1) (e.g. 61*21*11 = 14091)
        # self.fem.mesh.elemNodes: (numElems,8) array mapping each element to its 8 node indices (e.g. [12,13,34,33,145,146,167,166])
        # self.fem.mesh.elemArea: tensor of element volumes (or areas in 2D) (e.g. [1.0,1.0,...] for uniform grid)
        # self.fem.mesh.netArea: total mesh volume/area = sum(elemArea) (e.g. 12000.0)
        # self.fem.mesh.nodeXYZ: (numNodes,3) array of node coordinates [x,y,z] (e.g. [2.0,15.0,3.0])
        # self.fem.mesh.elemCenters: (numElems,3) coordinates of element centers (e.g. [2.5,14.5,3.5])
        # self.fem.mesh.elemCentersUpSampling: (numElems*8,3) coordinates of sub-element centers for higher resolution visualization
        # self.fem.mesh.bb_xmin: mesh bounding box minimum x-coordinate (e.g. 0)
        # self.fem.mesh.bb_xmax: mesh bounding box maximum x-coordinate (e.g. nelx*dx = 60)
        # self.fem.mesh.bb_ymin: mesh bounding box minimum y-coordinate (e.g. 0)
        # self.fem.mesh.bb_ymax: mesh bounding box maximum y-coordinate (e.g. nely*dy = 20)
        # self.fem.mesh.bb_zmin: mesh bounding box minimum z-coordinate (e.g. 0)
        # self.fem.mesh.bb_zmax: mesh bounding box maximum z-coordinate (e.g. nelz*dz = 10)
        # self.fem.mesh.bc: dictionary defining boundary conditions (e.g. {'numDOFPerNode':3,'fixed':[0,1,2],'force':f})
        # self.fem.mesh.ndof: total number of degrees of freedom = numNodes*numDOFPerNode (e.g. 14091*3 = 42273)
        # self.fem.mesh.fixed: array of constrained DOF indices (e.g. [0,1,2,3,4,5])
        # self.fem.mesh.free: array of unconstrained DOF indices used in solving (e.g. remaining DOFs after removing fixed)
        # self.fem.mesh.f: (ndof,1) force vector applied to system (e.g. f[100]= -1000.0 for downward load)
        # self.fem.mesh.numDOFPerElem: DOFs per element = 8 nodes * DOF per node (e.g. 8*3 = 24)
        # self.fem.mesh.edofMat: (numElems,24) mapping from element to global DOF indices (e.g. [36,37,38,...])
        # self.fem.mesh.iK: row indices used for assembling sparse global stiffness matrix (COO format)
        # self.fem.mesh.jK: column indices used for assembling sparse global stiffness matrix (COO format)
        # self.fem.mesh.nodeIdx: tuple indexing structure used for batched sparse matrix assembly
        # self.fem.mesh.nonNullElem: indices of elements that remain active (e.g. material elements in topology optimization)
        # self.fem.mesh.nullElem: indices of void elements (complement of nonNullElem)
        # self.fem.mesh.material: dictionary of material properties (e.g. {'E':1.0,'nu':0.3,'penal':3})
        # self.fem.mesh.Emax: Young’s modulus of solid material (e.g. 1.0)
        # self.fem.mesh.nu: Poisson’s ratio of material (e.g. 0.3)
        # self.fem.mesh.penal: SIMP penalization factor used in topology optimization (e.g. 3)
        # self.fem.mesh.KE: (numElems,24,24) element stiffness matrices for all elements
        # self.fem.mesh.B: (6,24) strain-displacement matrix used to compute strain from nodal displacements
        # self.fem.mesh.G: shear modulus computed from material properties = E/(2*(1+nu)) (e.g. 0.3846)
        # self.fem.mesh.C: (6,6) compliance matrix (inverse of elasticity matrix) defining material stress-strain relation
        # self.fem.Ksize: number of active/free DOFs after removing fixed DOFs from the global system (e.g. self.fem.mesh.ndof - len(self.fem.mesh.fixed))
        # self.fem.valid_mask: boolean torch mask selecting stiffness-matrix entries that do not involve fixed DOFs (e.g. True for free-free entries, False for fixed-related entries)
        # self.fem.new_row_indices: remapped row indices for the reduced sparse stiffness matrix using only free DOFs (e.g. [0, 0, 1, 1, 2, ...])
        # self.fem.new_col_indices: remapped column indices for the reduced sparse stiffness matrix using only free DOFs (e.g. [0, 1, 0, 1, 2, ...])
        # self.fem.f: reduced force vector containing only entries corresponding to free DOFs, stored as a torch tensor on the chosen device (e.g. tensor([0., -1000., 0.], device='cuda:0'))
        # self.fem.H8: anisotropic 8-node hexahedral material/stiffness helper object used to compute orientation-dependent element stiffness matrices
        # self.fem.H8.P: (6,6) precomputed anisotropic material coefficient matrix (e.g. default tensor([[0.00126103, ...], ...]))
        # self.fem.H8.Q: (6,) precomputed anisotropic material coefficient vector (e.g. default tensor([0.00283733, 0.0099734, 0.03632479, 0., 0., 0.]))
        # self.fem.H8.Ef: Young’s modulus in the fiber direction (e.g. 10.0)
        # self.fem.H8.Et: Young’s modulus in the transverse direction (e.g. 1.0)
        # self.fem.H8.nuf: Poisson’s ratio in the fiber direction (e.g. 0.30)
        # self.fem.H8.nut: Poisson’s ratio in the transverse direction (e.g. 0.30)
        # self.fem.H8.C_inv_np: (6,6) NumPy constitutive stiffness matrix for the anisotropic material (e.g. array([[10.96, ...], ...]))
        # self.fem.H8.C_inv: (6,6) torch constitutive stiffness matrix on the selected device (e.g. tensor([[10.96, ...]], device='cuda:0'))
        # self.fem.H8.int_weight: (27,) Gauss quadrature weights for 3x3x3 integration in the H8 element (e.g. tensor([0.0214, 0.0343, ...]))
        # self.fem.H8.B: (27,6,24) strain-displacement matrices at all 27 Gauss points (e.g. one 6x24 matrix per integration point)
        # self.fem.H8.NodeB: (6,24) strain-displacement matrix used at the element center for nodal/element strain evaluation
        # self.fem.H8.temp_C: (batch,6,6) rotated constitutive matrices saved from the latest angle2Ke() call (e.g. one per element in the batch)
        # self.fem.H8.T: (batch,6,6) rotation/transformation matrices saved from the latest angle2Ke() call (e.g. one per element in the batch) 