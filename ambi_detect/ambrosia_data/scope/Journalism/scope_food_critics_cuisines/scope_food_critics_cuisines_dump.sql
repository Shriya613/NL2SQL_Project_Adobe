BEGIN TRANSACTION;
CREATE TABLE Cuisines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cuisine_name TEXT);
INSERT INTO "Cuisines" VALUES(1,'Italian');
INSERT INTO "Cuisines" VALUES(2,'Mexican');
INSERT INTO "Cuisines" VALUES(3,'Indian');
INSERT INTO "Cuisines" VALUES(4,'Chinese');
INSERT INTO "Cuisines" VALUES(5,'Japanese');
CREATE TABLE Food_Critics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    critic_name TEXT,
    contact_email TEXT);
INSERT INTO "Food_Critics" VALUES(1,'John Doe','john.doe@example.com');
INSERT INTO "Food_Critics" VALUES(2,'Jane Smith','jane.smith@example.com');
INSERT INTO "Food_Critics" VALUES(3,'Tom Brown','tom.brown@example.com');
INSERT INTO "Food_Critics" VALUES(4,'Alice Johnson','alice.johnson@example.com');
INSERT INTO "Food_Critics" VALUES(5,'Bob Davis','bob.davis@example.com');
CREATE TABLE Food_Critics_Cuisines (
    food_critic_id INTEGER,
    cuisine_id INTEGER,
    FOREIGN KEY (food_critic_id) REFERENCES Food_Critics(id),
    FOREIGN KEY (cuisine_id) REFERENCES Cuisines(id));
INSERT INTO "Food_Critics_Cuisines" VALUES(2,1);
INSERT INTO "Food_Critics_Cuisines" VALUES(3,1);
INSERT INTO "Food_Critics_Cuisines" VALUES(4,1);
INSERT INTO "Food_Critics_Cuisines" VALUES(5,1);
INSERT INTO "Food_Critics_Cuisines" VALUES(1,2);
INSERT INTO "Food_Critics_Cuisines" VALUES(3,2);
INSERT INTO "Food_Critics_Cuisines" VALUES(4,2);
INSERT INTO "Food_Critics_Cuisines" VALUES(5,2);
INSERT INTO "Food_Critics_Cuisines" VALUES(1,3);
INSERT INTO "Food_Critics_Cuisines" VALUES(2,3);
INSERT INTO "Food_Critics_Cuisines" VALUES(4,3);
INSERT INTO "Food_Critics_Cuisines" VALUES(5,3);
INSERT INTO "Food_Critics_Cuisines" VALUES(1,4);
INSERT INTO "Food_Critics_Cuisines" VALUES(2,4);
INSERT INTO "Food_Critics_Cuisines" VALUES(3,4);
INSERT INTO "Food_Critics_Cuisines" VALUES(5,4);
INSERT INTO "Food_Critics_Cuisines" VALUES(1,5);
INSERT INTO "Food_Critics_Cuisines" VALUES(2,5);
INSERT INTO "Food_Critics_Cuisines" VALUES(3,5);
INSERT INTO "Food_Critics_Cuisines" VALUES(4,5);
INSERT INTO "Food_Critics_Cuisines" VALUES(5,5);
INSERT INTO "Food_Critics_Cuisines" VALUES(1,1);
INSERT INTO "Food_Critics_Cuisines" VALUES(2,2);
INSERT INTO "Food_Critics_Cuisines" VALUES(3,3);
INSERT INTO "Food_Critics_Cuisines" VALUES(4,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cuisines',5);
INSERT INTO "sqlite_sequence" VALUES('Food_Critics',5);
COMMIT;
