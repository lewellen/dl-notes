import numpy as np
import matplotlib.pyplot as plt

def generateScalarMesh(X, Y, f):
    Z = []
    for y in Y:
        for x in X:
            Z.append(f(np.array([x, y])))

    Z = np.array(Z).reshape((len(X), len(Y)))
    return Z

def generateVectorMesh(X, Y, f):
    U = []
    V = []
    for y in Y:
        for x in X:
            z = f(np.array([x, y]))
            U.append(z[0])
            V.append(z[1])
            
    U = np.array(U).reshape((len(X), len(Y)))
    V = np.array(V).reshape((len(X), len(Y)))
    return U, V

def marginedSupport(xs, marginPercent = 0.75, steps = 100):
    xmin = min(xs)
    xmax = max(xs)
    xrange_ = xmax - xmin
    xmargin = marginPercent * xrange_
    support = np.linspace(xmin - xmargin, xmax + xmargin, steps)
    return support

def contourPlot(fig, ax, X, Y, Z):
    cmap = plt.get_cmap('viridis')
    im = ax.contourf(X, Y, Z, cmap=cmap)
    fig.colorbar(im, ax=ax)
    
def streamPlot(fig, ax, trace, f):
    start = trace[0]
    end = trace[-1]
    
    xs = map(lambda x : x[0], trace)
    ys = map(lambda x : x[1], trace)

    X = marginedSupport(xs, 25)
    Y = marginedSupport(ys, 25)
    U, V = generateVectorMesh(X, Y, lambda x : -f(x))
    ax.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
        
def trajectoryPlot(fig, ax, trace, f):
    start = trace[0]
    end = trace[-1]
    
    xs = map(lambda x : x[0], trace)
    ys = map(lambda x : x[1], trace)

    X = marginedSupport(xs)
    Y = marginedSupport(ys)
    Z = generateScalarMesh(X, Y, lambda x : np.log(f(x)))

    contourPlot(fig, ax, X, Y, Z)
    
    ax.plot(xs, ys, c='r')
    ax.scatter(trace[0][0], trace[0][1], label="Start")
    ax.scatter(trace[-1][0], trace[-1][1], label="End")
    ax.legend()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("log(Objective Function) Contour")

    print "Start: f(", trace[0], ") =", f(trace[0])
    print "End: f(", trace[-1], ") =", f(trace[-1])
    
def convergencePlot(ax, trace, f, dfd0):
    ax.plot(range(0, len(trace)), map(lambda x : f(x), trace), label=r'f(x,y)')
    ax.plot(range(0, len(trace)), map(lambda x : np.linalg.norm(dfd0(x)), trace), label=r'norm(\nabla f)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log(Function Value)")
    ax.set_title("Convergence")
    ax.set_yscale("log")
    ax.legend()
