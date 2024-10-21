import sqlite3
import numpy as np

def read_features(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM features')
    rows = cursor.fetchall()
    conn.close()
    return rows

def display_features(rows):
    for row in rows:
        filename = row[1]
        area = row[2]
        volume = row[3]
        compactness = row[4]
        rectangularity = row[5]
        convexity = row[6]
        diameter = row[7]
        eccentricity = row[8]
        descriptor = np.frombuffer(row[9], dtype=np.float64)
        print(f"Filename: {filename}")
        print(f"Area: {area}")
        print(f"Volume: {volume}")
        print(f"Compactness: {compactness}")
        print(f"Rectangularity: {rectangularity}")
        print(f"Convexity: {convexity}")
        print(f"Diameter: {diameter}")
        print(f"Eccentricity: {eccentricity}")
        print(f"Descriptor: {descriptor}\n")

def main():
    db_name = "extracted_features/your_mesh_features.db"
    rows = read_features(db_name)
    display_features(rows)

if __name__ == "__main__":
   main()