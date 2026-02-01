BEGIN TRANSACTION;
CREATE TABLE Agroforestry_Systems(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_name TEXT,
    description TEXT
);
INSERT INTO "Agroforestry_Systems" VALUES(1,'Silvopasture','A system where trees, livestock, and forages coexist.');
INSERT INTO "Agroforestry_Systems" VALUES(2,'Alley Cropping','Trees are planted alongside crops to provide shade and reduce erosion.');
INSERT INTO "Agroforestry_Systems" VALUES(3,'Forest Gardening','A diverse, multi-layered plant community based on woodland ecosystems.');
INSERT INTO "Agroforestry_Systems" VALUES(4,'Riparian Buffer','Strips of vegetation along waterways to filter runoff.');
INSERT INTO "Agroforestry_Systems" VALUES(5,'Windbreak','Rows of trees used to block wind.');
CREATE TABLE Farmers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Farmers" VALUES(1,'John','Doe','john.doe@email.com');
INSERT INTO "Farmers" VALUES(2,'Jane','Smith','jane.smith@email.com');
INSERT INTO "Farmers" VALUES(3,'Bob','Johnson','bob.johnson@email.com');
INSERT INTO "Farmers" VALUES(4,'Emily','Williams','emily.williams@email.com');
INSERT INTO "Farmers" VALUES(5,'Tom','Brown','tom.brown@email.com');
CREATE TABLE Farms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farm_name TEXT,
    location TEXT
);
INSERT INTO "Farms" VALUES(1,'Sunny Acres','123 Main St, Ruralville');
INSERT INTO "Farms" VALUES(2,'Green Pastures','456 Elm St, Hometown');
INSERT INTO "Farms" VALUES(3,'Orchard Haven','789 Oak Lane, Fruitful City');
INSERT INTO "Farms" VALUES(4,'Meadow View','101 Pine St, Seedling County');
INSERT INTO "Farms" VALUES(5,'Timber Ridge','202 Cedar Ave, Lumbertown');
CREATE TABLE Species(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Species" VALUES(1,'Oak');
INSERT INTO "Species" VALUES(2,'Apple');
INSERT INTO "Species" VALUES(3,'Cherry');
INSERT INTO "Species" VALUES(4,'Peach');
INSERT INTO "Species" VALUES(5,'Plum');
CREATE TABLE System_Farms(
    agroforestry_system_id INTEGER,
    farm_id INTEGER,
    FOREIGN KEY(agroforestry_system_id) REFERENCES Agroforestry_Systems(id),
    FOREIGN KEY(farm_id) REFERENCES Farms(id),
    PRIMARY KEY(agroforestry_system_id, farm_id)
);
CREATE TABLE System_Species(
    agroforestry_system_id INTEGER,
    species_id INTEGER,
    FOREIGN KEY(agroforestry_system_id) REFERENCES Agroforestry_Systems(id),
    FOREIGN KEY(species_id) REFERENCES Species(id),
    PRIMARY KEY(agroforestry_system_id, species_id)
);
INSERT INTO "System_Species" VALUES(1,1);
INSERT INTO "System_Species" VALUES(2,1);
INSERT INTO "System_Species" VALUES(3,1);
INSERT INTO "System_Species" VALUES(4,1);
INSERT INTO "System_Species" VALUES(5,1);
INSERT INTO "System_Species" VALUES(5,4);
CREATE TABLE Works_On(
    farmer_id INTEGER,
    farm_id INTEGER,
    FOREIGN KEY(farmer_id) REFERENCES Farmers(id),
    FOREIGN KEY(farm_id) REFERENCES Farms(id),
    PRIMARY KEY(farmer_id, farm_id)
);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Species',5);
INSERT INTO "sqlite_sequence" VALUES('Agroforestry_Systems',5);
INSERT INTO "sqlite_sequence" VALUES('Farms',5);
INSERT INTO "sqlite_sequence" VALUES('Farmers',5);
COMMIT;
