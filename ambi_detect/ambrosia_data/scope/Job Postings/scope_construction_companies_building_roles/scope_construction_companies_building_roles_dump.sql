BEGIN TRANSACTION;
CREATE TABLE Building_Roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_value TEXT
);
INSERT INTO "Building_Roles" VALUES(1,'Site Supervisor');
INSERT INTO "Building_Roles" VALUES(2,'Project Manager');
INSERT INTO "Building_Roles" VALUES(3,'Estimator');
INSERT INTO "Building_Roles" VALUES(4,'Foreman');
INSERT INTO "Building_Roles" VALUES(5,'Laborer');
CREATE TABLE Construction_Companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    address TEXT
);
INSERT INTO "Construction_Companies" VALUES(1,'Company A','123 Main St');
INSERT INTO "Construction_Companies" VALUES(2,'Company B','456 Elm St');
INSERT INTO "Construction_Companies" VALUES(3,'Company C','789 Oak St');
INSERT INTO "Construction_Companies" VALUES(4,'Company D','001 Pine St');
INSERT INTO "Construction_Companies" VALUES(5,'Company E','999 Maple Ave');
CREATE TABLE Construction_Companies_Building_Roles (
    construction_company_id INTEGER,
    building_role_id INTEGER,
    FOREIGN KEY (construction_company_id) REFERENCES Construction_Companies(id),
    FOREIGN KEY (building_role_id) REFERENCES Building_Roles(role_id),
    PRIMARY KEY (construction_company_id, building_role_id)
);
INSERT INTO "Construction_Companies_Building_Roles" VALUES(1,1);
INSERT INTO "Construction_Companies_Building_Roles" VALUES(2,1);
INSERT INTO "Construction_Companies_Building_Roles" VALUES(3,1);
INSERT INTO "Construction_Companies_Building_Roles" VALUES(4,1);
INSERT INTO "Construction_Companies_Building_Roles" VALUES(5,1);
INSERT INTO "Construction_Companies_Building_Roles" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Construction_Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Building_Roles',5);
COMMIT;
