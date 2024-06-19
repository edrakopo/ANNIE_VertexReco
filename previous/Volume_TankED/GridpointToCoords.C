#include <iostream>

double** find_Coords(int Gridpoint) {
    // Create a grid for 10cm, in volume 1x1m
    int step = 10;
    int gridpoint[10][10][10];

    double** coords = new double*[Gridpoint];
    for (int i = 0; i < Gridpoint; i++) {
        coords[i] = new double[3];
    }

    int index = 0;
    for (int z = 0; z < 10; z++) {
        for (int r = 0; r < 10; r++) {
            for (int c = 0; c < 10; c++) {
                gridpoint[z][r][c] = index;
                // Find vtx coordinates for the given gridpoint
                coords[index][0] = -50 + c * step + 5;
                coords[index][1] = -50 + r * step + 5;
                coords[index][2] = -50 + z * step + 5;
                index++;
            }
        }
    }

    return coords;
}
void GridpointToCoords() {
    ofstream csvfile;
    csvfile.open ("gridpoint_coords.csv");
    int Gridpoint = 1000;
    double** coordinates = find_Coords(Gridpoint);

    for (int i = 0; i < Gridpoint; i++) {
        std::cout << "Coordinate " << i << ": ";
        std::cout << coordinates[i][0] << ", ";
        std::cout << coordinates[i][1] << ", ";
        std::cout << coordinates[i][2] << std::endl;

        csvfile<<coordinates[i][0]<<","<<coordinates[i][1]<< ", "<<coordinates[i][2]<<endl;
    }
 
   

}

