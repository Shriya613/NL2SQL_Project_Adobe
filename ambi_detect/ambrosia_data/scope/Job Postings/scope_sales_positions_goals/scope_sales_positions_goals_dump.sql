BEGIN TRANSACTION;
CREATE TABLE Goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_type TEXT
);
INSERT INTO "Goals" VALUES(1,'Quota');
INSERT INTO "Goals" VALUES(2,'Revenue Growth');
INSERT INTO "Goals" VALUES(3,'Customer Acquisition');
INSERT INTO "Goals" VALUES(4,'Market Penetration');
INSERT INTO "Goals" VALUES(5,'Product Launch');
CREATE TABLE Sales_Positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_name TEXT,
    salary_range REAL
);
INSERT INTO "Sales_Positions" VALUES(1,'Account Executive',60000.0);
INSERT INTO "Sales_Positions" VALUES(2,'Business Development Manager',70000.0);
INSERT INTO "Sales_Positions" VALUES(3,'Senior Account Manager',80000.0);
INSERT INTO "Sales_Positions" VALUES(4,'Inside Sales Representative',55000.0);
INSERT INTO "Sales_Positions" VALUES(5,'Territory Sales Manager',75000.0);
CREATE TABLE Sales_Positions_Goals (
    sales_position_id INTEGER,
    goal_id INTEGER,
    FOREIGN KEY (sales_position_id) REFERENCES Sales_Positions (id),
    FOREIGN KEY (goal_id) REFERENCES Goals (id),
    PRIMARY KEY (sales_position_id, goal_id)
);
INSERT INTO "Sales_Positions_Goals" VALUES(1,1);
INSERT INTO "Sales_Positions_Goals" VALUES(1,2);
INSERT INTO "Sales_Positions_Goals" VALUES(2,3);
INSERT INTO "Sales_Positions_Goals" VALUES(2,4);
INSERT INTO "Sales_Positions_Goals" VALUES(3,5);
INSERT INTO "Sales_Positions_Goals" VALUES(4,6);
INSERT INTO "Sales_Positions_Goals" VALUES(5,7);
INSERT INTO "Sales_Positions_Goals" VALUES(2,1);
INSERT INTO "Sales_Positions_Goals" VALUES(3,1);
INSERT INTO "Sales_Positions_Goals" VALUES(4,1);
INSERT INTO "Sales_Positions_Goals" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sales_Positions',5);
INSERT INTO "sqlite_sequence" VALUES('Goals',5);
COMMIT;
