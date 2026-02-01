BEGIN TRANSACTION;
CREATE TABLE "Age" (
    id INTEGER PRIMARY KEY,
    age REAL,
    country TEXT
);
INSERT INTO "Age" VALUES(1,30.2,'USA');
INSERT INTO "Age" VALUES(2,32.6,'Canada');
INSERT INTO "Age" VALUES(3,29.8,'UK');
INSERT INTO "Age" VALUES(4,30.7,'Australia');
INSERT INTO "Age" VALUES(5,28.9,'Germany');
CREATE TABLE Country(
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Country" VALUES(1,'USA');
INSERT INTO "Country" VALUES(2,'Canada');
INSERT INTO "Country" VALUES(3,'UK');
INSERT INTO "Country" VALUES(4,'Australia');
INSERT INTO "Country" VALUES(5,'Germany');
CREATE TABLE "Population_Group"(
    id INTEGER PRIMARY KEY,
    group_name TEXT,
    population REAL,
    median_age INTEGER,
    FOREIGN KEY(median_age) REFERENCES "Age"(id));
INSERT INTO "Population_Group" VALUES(1,'Young Adults',1000000.0,1);
INSERT INTO "Population_Group" VALUES(2,'Adults',2000000.0,2);
INSERT INTO "Population_Group" VALUES(3,'Elderly',1500000.0,3);
INSERT INTO "Population_Group" VALUES(4,'Children',500000.0,4);
INSERT INTO "Population_Group" VALUES(5,'Teenagers',250000.0,5);
CREATE TABLE Region(
    id INTEGER PRIMARY KEY,
    region_name TEXT,
    country_id INTEGER,
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "Region" VALUES(1,'Northeast',1);
INSERT INTO "Region" VALUES(2,'Southwest',1);
INSERT INTO "Region" VALUES(3,'Central',1);
INSERT INTO "Region" VALUES(4,'Atlantic',1);
INSERT INTO "Region" VALUES(5,'Pacific',1);
INSERT INTO "Region" VALUES(6,'Ontario',2);
INSERT INTO "Region" VALUES(7,'Quebec',2);
INSERT INTO "Region" VALUES(8,'Maritimes',2);
INSERT INTO "Region" VALUES(9,'Alberta',3);
INSERT INTO "Region" VALUES(10,'Scotland',3);
INSERT INTO "Region" VALUES(11,'Wales',3);
INSERT INTO "Region" VALUES(12,'England',3);
INSERT INTO "Region" VALUES(13,'Victoria',4);
INSERT INTO "Region" VALUES(14,'New South Wales',4);
INSERT INTO "Region" VALUES(15,'Tasmania',4);
INSERT INTO "Region" VALUES(16,'Queensland',4);
INSERT INTO "Region" VALUES(17,'West Germany',5);
INSERT INTO "Region" VALUES(18,'East Germany',5);
INSERT INTO "Region" VALUES(19,'Bavaria',5);
INSERT INTO "Region" VALUES(20,'Hesse',5);
COMMIT;
