# https://www.cs.cmu.edu/~baraff/papers/sig98.pdf
import argparse

import numpy as np

import taichi as ti


@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1)**2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # numbser of edges
        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.invMass = ti.field(ti.f32, self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f32,
                                  self.NE)  # Jacobian with respect to position
        self.rest_len = ti.field(ti.f32, self.NE)
        self.ks = 1000.0  # spring stiffness
        self.kd = 0.5  # damping constant
        self.kf = 1.0e5  # fix point stiffness

        self.gravity = ti.Vector([0.0, -2.0])
        self.init_pos()
        self.init_edges()
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(
            2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.DBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV,
                                                      2 * self.NV,
                                                      max_num_triplets=10000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV,
                                                      2 * self.NV,
                                                      max_num_triplets=10000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()
        self.fix_vertex = [self.N, self.NV - 1]
        self.Jf = ti.Matrix.field(2, 2, ti.f32, 2)  # fix constraint hessian

        # For conjugate gradient method
        self.v_next = ti.Vector.field(2, ti.f32, self.NV)
        self.Adv = ti.Vector.field(2, ti.f32, self.NV)
        self.b = ti.Vector.field(2, ti.f32, self.NV)
        self.r = ti.Vector.field(2, ti.f32, self.NV)
        self.p = ti.Vector.field(2, ti.f32, self.NV)
        self.Ap = ti.Vector.field(2, ti.f32, self.NV)

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([i, j]) / self.N * 0.5 + ti.Vector(
                [0.25, 0.25])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0, 0])
            self.invMass[k] = 1.0

    @ti.kernel
    def init_edges(self):
        pos, spring, N, rest_len = ti.static(self.pos, self.spring, self.N,
                                             self.rest_len)
        for i, j in ti.ndrange(N + 1, N):
            idx, idx1 = i * N + j, i * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx1 + 1])
            rest_len[idx] = (pos[idx1] - pos[idx1 + 1]).norm()
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            idx, idx1, idx2 = start + i + j * N, i * (N + 1) + j, i * (
                N + 1) + j + N + 1
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            idx, idx1, idx2 = start + i * N + j, i * (N + 1) + j, (i + 1) * (
                N + 1) + j + 1
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
        start = 2 * N * (N + 1) + N * N
        for i, j in ti.ndrange(N, N):
            idx, idx1, idx2 = start + i * N + j, i * (N + 1) + j + 1, (
                i + 1) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

    @ti.kernel
    def init_mass_sp(self, M: ti.linalg.sparse_matrix_builder()):
        for i in range(self.NV):
            mass = 1.0 / self.invMass[i]
            M[2 * i + 0, 2 * i + 0] += mass
            M[2 * i + 1, 2 * i + 1] += mass

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        for i in self.force:
            self.force[i] += self.gravity / self.invMass[i]

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos2 - pos1
            force = self.ks * (dis.norm() -
                               self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force
        # fix constraint force
        self.force[self.N] += self.kf * (self.initPos[self.N] -
                                         self.pos[self.N])
        self.force[self.NV - 1] += self.kf * (self.initPos[self.NV - 1] -
                                              self.pos[self.NV - 1])

    @ti.kernel
    def compute_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]],
                               [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            self.Jx[i] = (I - self.rest_len[i] * l *
                          (I - dxtdx * l**2)) * self.ks
        # fix point constraint force Jacobian
        self.Jf[0] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])
        self.Jf[1] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])

    @ti.kernel
    def assemble_K(self, K: ti.linalg.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]
                K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]
        for m, n in ti.static(ti.ndrange(2, 2)):
            K[2 * self.N + m, 2 * self.N + n] += self.Jf[0][m, n]
            K[2 * (self.NV - 1) + m, 2 * (self.NV - 1) + n] += self.Jf[1][m, n]

    @ti.kernel
    def directUpdatePosVel(self, h: ti.f32, v_next: ti.ext_arr()):
        for i in self.pos:
            self.vel[i] = ti.Vector([v_next[2 * i], v_next[2 * i + 1]])
            self.pos[i] += h * self.vel[i]

    def update_direct(self, h):
        self.compute_force()
        self.compute_Jacobians()
        # Assemble global system
        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()
        A = self.M - h**2 * K
        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(A)
        solver.factorize(A)

        vel = self.vel.to_numpy().reshape(2 * self.NV)
        force = self.force.to_numpy().reshape(2 * self.NV)
        b = h * force + self.M @ vel

        v_next = solver.solve(b)
        # flag = solver.info()
        # print("solver flag: ", flag)
        self.directUpdatePosVel(h, v_next)

    @ti.kernel
    def compute_RHS(self, h: ti.f32):
        #rhs = b = h * force + M @ v
        for i in range(self.NV):
            self.b[i] = h * self.force[i] + 1.0 / self.invMass[i] * self.vel[i]

    @ti.func
    def dot(self, v1, v2):
        result = 0.0
        for i in range(self.NV):
            result += v1[i][0] * v2[i][0]
            result += v1[i][1] * v2[i][1]
        return result

    @ti.func
    def A_mult_x(self, h, dst, src):
        coeff = -h**2
        for i in range(self.NV):
            dst[i] = 1.0 / self.invMass[i] * src[i]
        for i in range(self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            temp = self.Jx[i] @ (src[idx1] - src[idx2])
            dst[idx1] -= coeff * temp
            dst[idx2] += coeff * temp
        # fix constraint
        fix1, fix2 = self.N, self.NV - 1
        dst[fix1] -= coeff * self.kf * src[fix1]
        dst[fix2] -= coeff * self.kf * src[fix2]

    # conjugate gradient solving
    # https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    @ti.kernel
    def cg(self, h: ti.f32):
        for i in range(self.NV):
            self.v_next[i] = ti.Vector([0.0, 0.0])
        self.A_mult_x(h, self.Adv, self.v_next)  # Adv = A @ dv
        for i in range(self.NV):  # r = b - A * dv
            self.r[i] = self.b[i] - self.Adv[i]
        for i in range(self.NV):  # d = r
            self.p[i] = self.r[i]
        epsNew = self.dot(self.r, self.r)
        ite, iteMax = 0, 2 * self.NV
        while ite < iteMax and epsNew > 1.0e-6:
            self.A_mult_x(h, self.Ap, self.p)  # Ap = A @ p
            alpha = epsNew / self.dot(
                self.p, self.Ap)  # alpha = (r^T * r) / dot(p, Ap)
            for i in range(self.NV):
                self.v_next[i] += alpha * self.p[i]  # x^{i+} += alpha * d
                self.r[i] -= alpha * self.Ap[i]  # r^{i+1} -= alpha * Ap
            epsOld = epsNew
            epsNew = self.dot(self.r, self.r)
            beta = epsNew / epsOld
            for i in range(self.NV):
                self.p[i] = self.r[i] + beta * self.p[
                    i]  #p^{i+1} = r^{i+1} + beta * p^{i}
            ite += 1

    @ti.kernel
    def cgUpdatePosVel(self, h: ti.f32):
        for i in self.pos:
            self.vel[i] = self.v_next[i]
            self.pos[i] += h * self.vel[i]

    def update_cg(self, h):
        self.compute_force()
        self.compute_Jacobians()
        self.compute_RHS(h)
        self.cg(h)
        self.cgUpdatePosVel(h)

    def display(self, gui, radius=5, color=0xffffff):
        lines = self.spring.to_numpy()
        pos = self.pos.to_numpy()
        edgeBegin = np.zeros(shape=(lines.shape[0], 2))
        edgeEnd = np.zeros(shape=(lines.shape[0], 2))
        for i in range(lines.shape[0]):
            idx1, idx2 = lines[i][0], lines[i][1]
            edgeBegin[i] = pos[idx1]
            edgeEnd[i] = pos[idx2]
        gui.lines(edgeBegin, edgeEnd, radius=2, color=0x0000ff)
        gui.circles(self.pos.to_numpy(), radius, color)

    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[2 * i + 0] = self.spring[i][0]
            self.indices[2 * i + 1] = self.spring[i][1]

    def displayGGUI(self, canvas, radius=0.01, color=(1.0, 1.0, 1.0)):
        self.spring2indices()
        canvas.lines(self.pos,
                     width=0.005,
                     indices=self.indices,
                     color=(0.0, 0.0, 1.0))
        canvas.circles(self.pos, radius, color)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    h = 0.01
    cloth = Cloth(N=10)
    pause = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-cg',
                        '--use_cg',
                        action='store_true',
                        help='Solve Ax=b with conjugate gradient method (CG).')
    args, unknowns = parser.parse_known_args()
    use_cg = False
    use_cg = args.use_cg
    gui = ti.GUI('Implicit Mass Spring System', res=(500, 500))
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                pause = not pause
        if not pause:
            if use_cg:
                cloth.update_cg(h)
            else:
                cloth.update_direct(h)
        cloth.display(gui)

        gui.show()
