#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

#define FILE_NAME "political.txt"
#define K 2

struct DistClass {
    int classif;
    float distance;
};

bool distClassCompare(DistClass d1, DistClass d2) {
    return d1.distance < d2.distance;
}

// Relatively quick function because K is usually small.
// Meaning, parallelization would yield negligible performance
// increases.
int chooseMajority(DistClass *vals, unsigned int length, int numClass) {
    int *histogram = new int[numClass];
    
    // Initialize the histogram
    for (int i = 0; i < numClass; i++) {
        histogram[i] = 0;
    }
    
    // Count the values.
    for (int i = 0; i < K; i++) {
        // Make sure we're not above array bounds
        if (i < length) {
            histogram[vals[i].classif]++;
        }
    }
    
    /*for (int i = 0; i < numClass; i++) {
        cout << i << " " << histogram[i] << endl;
    }*/
    
    // Find the element of the majority
    int maxClass = distance(histogram, max_element(histogram, histogram + numClass));
    return maxClass;
}

void sortDistances(DistClass *vals, unsigned int length) {
  std::sort(vals, vals + length, distClassCompare);
  return;
}

//find distance
DistClass* find_distance (float* unknownObject, float** knownSamples, int* classif, 
    int numAtributes, int numKnownSamples) 
{
    
    DistClass* distances;
    distances = (DistClass *) malloc(sizeof(struct DistClass) * numKnownSamples);
    
    for (int i = 0; i < numKnownSamples; i++)
    {
        
        distances[i].classif = classif[i];
        
        float distance = 0;
        for (int j = 0; j < numAtributes; j++)
        {
            float x = unknownObject[j];
            float t = knownSamples[i][j];
            
            distance = distance + ((x-t) * (x-t));
        }
        
        distance = sqrt(distance);
        distances[i].distance = distance;
        
    }
    
    return distances;
}
    

void parse(int* numAttributes, int* numKnownSamples, int* numClass, int *numUnknowns,
    float ** min, float ** max, float *** knowns, int ** classifications, 
    float *** unknowns, string** unknownNames)
{
    ifstream myfile(FILE_NAME, ios::in);  // declare and open
    
    int numAttrib, numKnownSamp, numCla, numUn;
    myfile >> numKnownSamp >> numAttrib >> numCla >> numUn;
    
    *numAttributes = numAttrib;
    *numKnownSamples = numKnownSamp;
    *numClass = numCla;
    *numUnknowns = numUn;
    
    // Populate all of the mins and maxes
    *min = (float*) malloc(sizeof(float) * numAttrib);
    *max = (float*) malloc(sizeof(float) * numAttrib);
    for (int i = 0; i < numAttrib; i++) {
        float currentMax, currentMin;
        myfile >> currentMin >> currentMax;
        (*min)[i] = currentMin;
        (*max)[i] = currentMax;
    }
    
    
    // Populate the known object types
    *classifications =(int*) malloc(sizeof(int) * numKnownSamp);
    *knowns = (float**) malloc(sizeof(float*) * numKnownSamp);
    
    for (int i = 0; i < numKnownSamp; i++) {
        (*knowns)[i] = (float*) malloc(sizeof(float) * numAttrib);
        int currentClass;
        myfile >> currentClass;
        (*classifications)[i] = currentClass;
        
        for (int j = 0; j < numAttrib; j++) {
            float currentAttrib;
            myfile >> currentAttrib;
            (*knowns)[i][j] = currentAttrib;
        }
    }
    
    
    // Populate the unknown object types
    *unknownNames = new string[numUn];
    *unknowns = (float**) malloc(sizeof(float*) * numUn);
    
    
    for (int i = 0; i < numUn; i++) {
        (*unknowns)[i] = (float*) malloc(sizeof(float) * numAttrib);
        string currentName;
        myfile >> currentName;
        (*unknownNames)[i] = currentName;
        
        for (int j = 0; j < numAttrib; j++) {
            float currentAttrib;
            myfile >> currentAttrib;
            (*unknowns)[i][j] = currentAttrib;
        }
    }
    
    myfile.close();
}


void normalize(float *min, float *max, float **knowns, int numAttributes, int numKnownSamples){
    for(int i = 0; i < numKnownSamples; i++){
        for(int i2 = 0; i2 < numAttributes; i2++ ){
            knowns[i][i2] =  (knowns[i][i2] - min[i2])/(max[i2]-min[i2]);
        }
    }
}

void outputParse(int numAttributes, int numKnownSamples, int numClass, 
    int numUnknowns, float* min, float* max, int* classifications, float** knowns,
    string* unknownNames, float** unknowns) {
    cout << numAttributes << " " << numKnownSamples <<  " " << numClass << " " 
        << numUnknowns << endl;
        
    for (int i = 0; i < numAttributes; i++) { 
        cout << "min/max: " << min[i] << " " << max[i] << endl;   
    }
    
    for (int i = 0; i < numKnownSamples; i++) {
        cout << classifications[i] << " ";
        for (int j = 0; j < numAttributes; j++) {
            cout << knowns[i][j] << " ";
        }
        cout << "\n";
    }
    
    for (int i = 0; i < numUnknowns; i++) {
        cout << unknownNames[i] << " ";
        for (int j = 0; j < numAttributes; j++) {
            cout << unknowns[i][j] << " ";
        }
        cout << "\n";
    }
}

int main() {
    int numAttributes, numKnownSamples, numClass, numUnknowns;
    float *min, *max;
    float **knowns;
    int *classifications;
    float **unknowns;
    string *unknownNames;
    std::clock_t start;
    
    parse(&numAttributes, &numKnownSamples, &numClass, &numUnknowns, 
        &min, &max, &knowns, &classifications, &unknowns, &unknownNames);
    
    start = std::clock();
    
    // Normalize both the knowns and unknown arrays
    normalize(min, max, knowns, numAttributes, numKnownSamples);
    normalize(min, max, unknowns, numAttributes, numUnknowns);
    
    // Find the distances from all of the unknown objects and the known
    // objects.
   for (int cUn = 0; cUn < numUnknowns; cUn++) {
        DistClass* dc = find_distance (unknowns[cUn], knowns, classifications, 
            numAttributes, numKnownSamples);
        //for (int i = 0; i < numKnownSamples; i++) {
        //    cout << dc[i].distance << " "; 
        //} cout << endl;
        
        
        //for (int i = 0; i < numKnownSamples; i++) {
        //    cout << dc[i].classif << " " << dc[i].distance << endl;
        //}
        
        // Sort the distance objects
        sortDistances(dc, numKnownSamples);
        /*for (int i = 0; i < numKnownSamples; i++) {
            cout << dc[i].classif << " " << dc[i].distance << endl;
        }*/
        
        //chooseMajority(dc, numKnownSamples, numClass);
        cout << unknownNames[cUn] << " " 
            << chooseMajority(dc, numKnownSamples, numClass) << endl;
            
   }
   
   duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
   
   std::cout<<"printf: "<< duration <<'\n';
}
