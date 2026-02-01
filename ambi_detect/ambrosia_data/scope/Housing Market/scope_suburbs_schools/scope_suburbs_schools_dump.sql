BEGIN TRANSACTION;
CREATE TABLE houses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    price DECIMAL(10,2),
    size INTEGER,
    bedrooms INTEGER,
    bathrooms INTEGER,
    suburb_id INTEGER,
    FOREIGN KEY (suburb_id) REFERENCES suburbs (id)
);
INSERT INTO "houses" VALUES(1,'1 Main St, SuburbA',500000,2000,3,2,1);
INSERT INTO "houses" VALUES(2,'2 Side St, SuburbB',450000,1800,3,2,2);
INSERT INTO "houses" VALUES(3,'3 Back Alley, SuburbC',400000,1600,2,2,3);
INSERT INTO "houses" VALUES(4,'4 Forward Ave, SuburbD',350000,1400,2,2,4);
INSERT INTO "houses" VALUES(5,'5 Overlook Blvd, SuburbE',300000,1200,2,2,5);
CREATE TABLE neighborhoods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    crime_rate REAL 
);
INSERT INTO "neighborhoods" VALUES(1,'Safe Neighborhood',1.5);
INSERT INTO "neighborhoods" VALUES(2,'Moderate Crime Area',3.0);
INSERT INTO "neighborhoods" VALUES(3,'High Crime Zone',4.5);
INSERT INTO "neighborhoods" VALUES(4,'Very High Crime Region',5.0);
INSERT INTO "neighborhoods" VALUES(5,'Low Crime Community',2.0);
CREATE TABLE schools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_type TEXT);
INSERT INTO "schools" VALUES(1,'Elementary School');
INSERT INTO "schools" VALUES(2,'High School');
INSERT INTO "schools" VALUES(3,'Middle School');
INSERT INTO "schools" VALUES(4,'College Prep School');
INSERT INTO "schools" VALUES(5,'Vocational School');
CREATE TABLE suburbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    population INTEGER
);
INSERT INTO "suburbs" VALUES(1,'SuburbA',10000);
INSERT INTO "suburbs" VALUES(2,'SuburbB',8000);
INSERT INTO "suburbs" VALUES(3,'SuburbC',7000);
INSERT INTO "suburbs" VALUES(4,'SuburbD',6000);
INSERT INTO "suburbs" VALUES(5,'SuburbE',5000);
CREATE TABLE suburbs_neighborhoods (
    suburb_id INTEGER,
    neighborhood_id INTEGER,
    FOREIGN KEY (suburb_id) REFERENCES suburbs (id),
    FOREIGN KEY (neighborhood_id) REFERENCES neighborhoods (id),
    PRIMARY KEY (suburb_id, neighborhood_id)
);
INSERT INTO "suburbs_neighborhoods" VALUES(1,1);
INSERT INTO "suburbs_neighborhoods" VALUES(1,2);
INSERT INTO "suburbs_neighborhoods" VALUES(2,3);
INSERT INTO "suburbs_neighborhoods" VALUES(2,4);
INSERT INTO "suburbs_neighborhoods" VALUES(3,5);
CREATE TABLE suburbs_schools (
    suburb_id INTEGER,
    school_id INTEGER,
    FOREIGN KEY (suburb_id) REFERENCES suburbs (id),
    FOREIGN KEY (school_id) REFERENCES schools (id)
);
INSERT INTO "suburbs_schools" VALUES(1,1);
INSERT INTO "suburbs_schools" VALUES(1,2);
INSERT INTO "suburbs_schools" VALUES(2,3);
INSERT INTO "suburbs_schools" VALUES(2,4);
INSERT INTO "suburbs_schools" VALUES(3,5);
INSERT INTO "suburbs_schools" VALUES(2,1);
INSERT INTO "suburbs_schools" VALUES(3,1);
INSERT INTO "suburbs_schools" VALUES(4,1);
INSERT INTO "suburbs_schools" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('suburbs',5);
INSERT INTO "sqlite_sequence" VALUES('schools',5);
INSERT INTO "sqlite_sequence" VALUES('houses',5);
INSERT INTO "sqlite_sequence" VALUES('neighborhoods',5);
COMMIT;
