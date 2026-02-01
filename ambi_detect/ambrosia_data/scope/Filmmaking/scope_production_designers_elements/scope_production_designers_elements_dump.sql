BEGIN TRANSACTION;
CREATE TABLE Elements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    element_type TEXT,
    description TEXT
);
INSERT INTO "Elements" VALUES(1,'Set_Design','Creates the visual environment for a film scene.');
INSERT INTO "Elements" VALUES(2,'Costume_Design','Designs and creates costumes for characters.');
INSERT INTO "Elements" VALUES(3,'Art_Direction','Oversees all aspects of the visual style of a film.');
INSERT INTO "Elements" VALUES(4,'Props','Objects used by actors during scenes.');
INSERT INTO "Elements" VALUES(5,'Location_Scouting','Finds suitable locations for shooting scenes.');
CREATE TABLE Production_Designers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Production_Designers" VALUES(1,'John Smith','johnsmith@example.com');
INSERT INTO "Production_Designers" VALUES(2,'Jane Doe','janedoe@example.com');
INSERT INTO "Production_Designers" VALUES(3,'Michael Johnson','michaeljohnson@example.com');
INSERT INTO "Production_Designers" VALUES(4,'Emily Davis','emilydavis@example.com');
INSERT INTO "Production_Designers" VALUES(5,'William Brown','williambrown@example.com');
CREATE TABLE Production_Designers_Elements (
    production_designer_id INTEGER,
    element_id INTEGER,
    FOREIGN KEY(production_designer_id) REFERENCES Production_Designers(id),
    FOREIGN KEY(element_id) REFERENCES Elements(id),
    PRIMARY KEY(production_designer_id, element_id)
);
INSERT INTO "Production_Designers_Elements" VALUES(1,1);
INSERT INTO "Production_Designers_Elements" VALUES(2,3);
INSERT INTO "Production_Designers_Elements" VALUES(3,1);
INSERT INTO "Production_Designers_Elements" VALUES(4,2);
INSERT INTO "Production_Designers_Elements" VALUES(5,1);
INSERT INTO "Production_Designers_Elements" VALUES(2,1);
INSERT INTO "Production_Designers_Elements" VALUES(4,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Production_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Elements',5);
COMMIT;
