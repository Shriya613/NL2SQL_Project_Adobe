BEGIN TRANSACTION;
CREATE TABLE Construction_Companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    location TEXT
);
INSERT INTO "Construction_Companies" VALUES(1,'Company A','New York');
INSERT INTO "Construction_Companies" VALUES(2,'Company B','Los Angeles');
INSERT INTO "Construction_Companies" VALUES(3,'Company C','Chicago');
INSERT INTO "Construction_Companies" VALUES(4,'Company D','Houston');
INSERT INTO "Construction_Companies" VALUES(5,'Company E','Miami');
CREATE TABLE Construction_Companies_Projects (
    construction_company_id INTEGER,
    project_id INTEGER,
    FOREIGN KEY(construction_company_id) REFERENCES Construction_Companies(id),
    FOREIGN KEY(project_id) REFERENCES Projects(id),
    PRIMARY KEY(construction_company_id, project_id)
);
INSERT INTO "Construction_Companies_Projects" VALUES(1,1);
INSERT INTO "Construction_Companies_Projects" VALUES(2,1);
INSERT INTO "Construction_Companies_Projects" VALUES(3,1);
INSERT INTO "Construction_Companies_Projects" VALUES(4,1);
INSERT INTO "Construction_Companies_Projects" VALUES(5,1);
INSERT INTO "Construction_Companies_Projects" VALUES(5,2);
CREATE TABLE House_Sizes (
    house_size_id INTEGER PRIMARY KEY AUTOINCREMENT,
    size_in_sqft INTEGER,
    price PERCENTAGE
);
INSERT INTO "House_Sizes" VALUES(1,1500,200);
INSERT INTO "House_Sizes" VALUES(2,2000,250);
INSERT INTO "House_Sizes" VALUES(3,2500,300);
INSERT INTO "House_Sizes" VALUES(4,3000,350);
INSERT INTO "House_Sizes" VALUES(5,3500,400);
CREATE TABLE Material_Types (
    material_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_of_material TEXT,
    cost_per_unit FLOAT
);
INSERT INTO "Material_Types" VALUES(1,'Concrete',50.0);
INSERT INTO "Material_Types" VALUES(2,'Steel',80.0);
INSERT INTO "Material_Types" VALUES(3,'Wood',40.0);
INSERT INTO "Material_Types" VALUES(4,'Aluminum',90.0);
INSERT INTO "Material_Types" VALUES(5,'Glass',70.0);
CREATE TABLE Projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_type TEXT,
    start_date DATE,
    end_date DATE
);
INSERT INTO "Projects" VALUES(1,'Foundation','2023-01-01','2023-06-01');
INSERT INTO "Projects" VALUES(2,'Residential','2023-07-01','2023-12-01');
INSERT INTO "Projects" VALUES(3,'Commercial','2024-01-01','2024-12-01');
INSERT INTO "Projects" VALUES(4,'Industrial','2025-01-01','2025-12-01');
INSERT INTO "Projects" VALUES(5,'Infrastructure','2026-01-01','2026-12-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Construction_Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
INSERT INTO "sqlite_sequence" VALUES('House_Sizes',5);
INSERT INTO "sqlite_sequence" VALUES('Material_Types',5);
COMMIT;
