#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NX      200   // Nombre de nœuds en x (longueur du canal)
#define NY      50    // Nombre de nœuds en y (hauteur du canal)
#define NSTEPS  50000 // Nombre total de pas de temps simulés
#define NSAVE   1000  // Intervalle entre deux sauvegardes (en pas de temps)

#define TAU     0.6   // Temps de relaxation BGK ; viscosité nu = (TAU-0.5)/3
#define RHO_IN  1.005 // Densité imposée à l'entrée
#define RHO_OUT 1.000 // Densité imposée à la sortie
#define UX_IN   0.01  // Vitesse uniforme imposée à l'entrée (cas 2 uniquement)

#define Q 9


// #define CAS_PRESSION   // Cas 1 : pression entrée / pression sortie
#define CAS_VITESSE // Cas 2 : vitesse entrée  / pression sortie

static const int cx[Q] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
static const int cy[Q] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

static const double w[Q] = {
    4.0/9.0,
    1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

static const int opp[Q] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

static double f[NX][NY][Q];    // Populations courantes
static double ftmp[NX][NY][Q]; // Populations post-collision (tampon)

static inline double feq(int k, double rho, double ux, double uy)
{
    double cu = cx[k]*ux + cy[k]*uy; // Projection de la vitesse sur la direction k
    double u2 = ux*ux + uy*uy;       // Carré de la norme de vitesse
    return w[k] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
}

static inline void macro(int x, int y, double *rho, double *ux, double *uy)
{
    double r = 0.0, mx = 0.0, my = 0.0;
    for (int k = 0; k < Q; k++) {
        r  += f[x][y][k];
        mx += cx[k] * f[x][y][k];
        my += cy[k] * f[x][y][k];
    }
    *rho = r;
    *ux  = mx / r;
    *uy  = my / r;
}


// Création de colormap
// Convertit t de [0,1] → couleur RGB (bleu=lent, rouge=rapide)

static void colormap_jet(double t, unsigned char *R, unsigned char *G, unsigned char *B)
{
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    double r, g, b;
    if (t < 0.125) {
        r = 0.0; g = 0.0; b = 0.5 + t * 4.0;
    } else if (t < 0.375) {
        r = 0.0; g = (t - 0.125) * 4.0; b = 1.0;
    } else if (t < 0.625) {
        r = (t - 0.375) * 4.0; g = 1.0; b = 1.0 - (t - 0.375) * 4.0;
    } else if (t < 0.875) {
        r = 1.0; g = 1.0 - (t - 0.625) * 4.0; b = 0.0;
    } else {
        r = 1.0 - (t - 0.875) * 4.0; g = 0.0; b = 0.0;
    }
    *R = (unsigned char)(r * 255.0);
    *G = (unsigned char)(g * 255.0);
    *B = (unsigned char)(b * 255.0);
}

// Création de VTK

static void save_vtk(int step)
{
    char fname[64];
    snprintf(fname, sizeof(fname), "velmap_%06d.vtk", step);
    FILE *fp = fopen(fname, "w");
    if (!fp) { perror("fopen vtk"); return; }

    // En-tête VTK obligatoire
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "LBM2D t=%d\n", step);          // Description libre
    fprintf(fp, "ASCII\n");                      // Format texte (lisible sans outil)
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");  // Grille cartésienne régulière

    // Dimensions de la grille (NX × NY × 1 car 2D)
    fprintf(fp, "DIMENSIONS %d %d 1\n", NX, NY);
    fprintf(fp, "ORIGIN 0 0 0\n");               // Coin bas-gauche du domaine
    fprintf(fp, "SPACING 1 1 1\n");              // Espacement entre nœuds (unités réseau)

    fprintf(fp, "POINT_DATA %d\n", NX * NY);     // Nombre total de points

    // Champ scalaire : densité rho
    fprintf(fp, "SCALARS rho double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int y = 0; y < NY; y++) {               // VTK : ordre y d'abord, puis x
        for (int x = 0; x < NX; x++) {
            double rho, ux, uy;
            macro(x, y, &rho, &ux, &uy);
            fprintf(fp, "%.6e\n", rho);
        }
    }

    // Champ scalaire : vitesse en x
    fprintf(fp, "SCALARS ux double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            double rho, ux, uy;
            macro(x, y, &rho, &ux, &uy);
            fprintf(fp, "%.6e\n", ux);
        }
    }

    // Champ scalaire : vitesse en y
    fprintf(fp, "SCALARS uy double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            double rho, ux, uy;
            macro(x, y, &rho, &ux, &uy);
            fprintf(fp, "%.6e\n", uy);
        }
    }

    // Champ vectoriel : vitesse (ux, uy, 0) — la composante z est nulle (2D)
    fprintf(fp, "VECTORS velocity double\n");
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            double rho, ux, uy;
            macro(x, y, &rho, &ux, &uy);
            fprintf(fp, "%.6e %.6e 0.0\n", ux, uy); // z=0 pour un écoulement 2D
        }
    }

    fclose(fp);
    printf("  VTK    : %s\n", fname);
}

// Initialisation

static void init(void)
{
    for (int x = 0; x < NX; x++) {
        // Gradient de densité linéaire : profil de pression stationnaire en entrée
        double rho0 = RHO_OUT + (RHO_IN - RHO_OUT) * (NX - 1 - x) / (double)(NX - 1);
        for (int y = 0; y < NY; y++)
            for (int k = 0; k < Q; k++)
                f[x][y][k] = feq(k, rho0, 0.0, 0.0); // Vitesse initiale nulle
    }
}

// Collision

static void collision(void)
{
    double omega = 1.0 / TAU; // Fréquence de relaxation
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            double rho, ux, uy;
            macro(x, y, &rho, &ux, &uy);
            for (int k = 0; k < Q; k++)
                ftmp[x][y][k] = f[x][y][k]
                               - omega * (f[x][y][k] - feq(k, rho, ux, uy));
        }
    }
}

// Propagation

static void streaming(void)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int k = 0; k < Q; k++) {
                int xn = x + cx[k];
                int yn = y + cy[k];
                if (xn < 0)   xn = NX - 1; // Périodicité (sera écrasée par les CL)
                if (xn >= NX) xn = 0;
                if (yn >= 0 && yn < NY)
                    f[xn][yn][k] = ftmp[x][y][k];
            }
        }
    }
}

// Parois No-Slip

static void bc_walls(void)
{
    for (int x = 0; x < NX; x++) {
        for (int k = 0; k < Q; k++) {
            if (cy[k] > 0) f[x][0][k]    = ftmp[x][0][opp[k]];    // Paroi basse y=0
            if (cy[k] < 0) f[x][NY-1][k] = ftmp[x][NY-1][opp[k]]; // Paroi haute y=NY-1
        }
    }
}


// CAS 1 : Pression en entrée - Pression en sortie (Zou-He) - Impose rho=RHO_IN à gauche, rho=RHO_OUT à droite

static void bc_inlet_pression(void)
{
    int x = 0;
    for (int y = 1; y < NY - 1; y++) {
        double rho = RHO_IN; // Densité imposée
        double ux  = 1.0 - (f[x][y][0] + f[x][y][2] + f[x][y][4] + 2.0*(f[x][y][3] + f[x][y][6] + f[x][y][7])) / rho;
        f[x][y][1] = f[x][y][3] + (2.0/3.0)*rho*ux;
        f[x][y][5] = f[x][y][7] - 0.5*(f[x][y][2] - f[x][y][4]) + (1.0/6.0)*rho*ux;
        f[x][y][8] = f[x][y][6] + 0.5*(f[x][y][2] - f[x][y][4]) + (1.0/6.0)*rho*ux;
    }
}


// CAS 1 & 2 : Pression en sortie (Zou-He) — identique pour les deux cas

static void bc_outlet_pression(void)
{
    int x = NX - 1;
    for (int y = 1; y < NY - 1; y++) {
        double rho = RHO_OUT; // Densité imposée
        double ux  = -1.0 + (f[x][y][0] + f[x][y][2] + f[x][y][4] + 2.0*(f[x][y][1] + f[x][y][5] + f[x][y][8])) / rho;
        f[x][y][3] = f[x][y][1] - (2.0/3.0)*rho*ux;
        f[x][y][7] = f[x][y][5] + 0.5*(f[x][y][2] - f[x][y][4]) - (1.0/6.0)*rho*ux;
        f[x][y][6] = f[x][y][8] - 0.5*(f[x][y][2] - f[x][y][4]) - (1.0/6.0)*rho*ux;
    }
}


// CAS 2 : Vitesse en entrée - Pression en sortie (Zou-He) - Impose ux=UX_IN uniforme à gauche.
// La densité est inconnue et déduite des populations connues.

static void bc_inlet_vitesse(void)
{
    int x = 0;
    for (int y = 1; y < NY - 1; y++) {
        double ux = UX_IN; // Vitesse uniforme imposée 
        // Densité inconnue à l'entrée : calculée à partir des populations connues + ux imposé
        double rho = (f[x][y][0] + f[x][y][2] + f[x][y][4] + 2.0*(f[x][y][3] + f[x][y][6] + f[x][y][7])) / (1.0 - ux);
        f[x][y][1] = f[x][y][3] + (2.0/3.0)*rho*ux;
        f[x][y][5] = f[x][y][7] - 0.5*(f[x][y][2] - f[x][y][4]) + (1.0/6.0)*rho*ux;
        f[x][y][8] = f[x][y][6] + 0.5*(f[x][y][2] - f[x][y][4]) + (1.0/6.0)*rho*ux;
    }
}

// Coins : copie du voisin intérieur diagonal (commune aux deux cas)
static void bc_corners(void)
{
    for (int k = 0; k < Q; k++) {
        f[0][0][k] = f[1][1][k];
        f[0][NY-1][k] = f[1][NY-2][k];
        f[NX-1][0][k] = f[NX-2][1][k];
        f[NX-1][NY-1][k] = f[NX-2][NY-2][k];
    }
}


// Sauvegarde .dat 

static void save_dat(int step)
{
    char fname[64];
    snprintf(fname, sizeof(fname), "output_%06d.dat", step);
    FILE *fp = fopen(fname, "w");
    if (!fp) { perror("fopen dat"); return; }
    fprintf(fp, "# x y ux uy rho\n");
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            double rho, ux, uy;
            macro(x, y, &rho, &ux, &uy);
            fprintf(fp, "%d %d %.6e %.6e %.6e\n", x, y, ux, uy, rho);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("  Données : %s\n", fname);
}

int main(void)
{
    printf("LBM 2D  -  D2Q9  -  Canal de Poiseuille\n");
    printf("Domaine : %d x %d noeuds\n", NX, NY);
    printf("tau=%.3f  nu=%.4f\n", TAU, (TAU - 0.5) / 3.0);

#if defined(CAS_PRESSION)
    printf("Cas : pression entree (rho=%.4f) / pression sortie (rho=%.4f)\n\n",
           RHO_IN, RHO_OUT);
#elif defined(CAS_VITESSE)
    printf("Cas : vitesse entree (ux=%.4f) / pression sortie (rho=%.4f)\n\n",
           UX_IN, RHO_OUT);
#endif

    init();

    for (int t = 1; t <= NSTEPS; t++) {
        collision();
        streaming();
        bc_walls();

#if defined(CAS_PRESSION)
        bc_inlet_pression();
        bc_outlet_pression();
#elif defined(CAS_VITESSE)
        bc_inlet_vitesse();
        bc_outlet_pression(); // La sortie en pression est identique dans les deux cas
#endif
        bc_corners();

        if (t % NSAVE == 0) {
            printf("t=%6d\n", t);
            save_dat(t);  // Fichier texte
            save_vtk(t);  // Fichier VTK pour ParaView (ux, uy, rho, vecteurs vitesse)
        }
    }

    printf("\nSimulation terminée.\n");
    return 0;
}
