BEGIN TRANSACTION;
CREATE TABLE Companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Companies" VALUES(1,'TechCorp');
INSERT INTO "Companies" VALUES(2,'Software Inc.');
INSERT INTO "Companies" VALUES(3,'Data Solutions Ltd.');
INSERT INTO "Companies" VALUES(4,'AI Developers');
INSERT INTO "Companies" VALUES(5,'Network Services Co.');
CREATE TABLE Customer_Facing_Roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT
);
INSERT INTO "Customer_Facing_Roles" VALUES(1,'Customer Service Advisor');
INSERT INTO "Customer_Facing_Roles" VALUES(2,'IT Support Technician');
CREATE TABLE Employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    phone TEXT,
    customer_role_id INTEGER,
    language_proficiency_id INTEGER, FullName TEXT,
    FOREIGN KEY(customer_role_id) REFERENCES Customer_Facing_Roles(id),
    FOREIGN KEY(language_proficiency_id) REFERENCES Language_Proficiencies(id)
);
INSERT INTO "Employees" VALUES(1,'mikeadams@email.com','+1112223333',1,1,'Mike Adams');
INSERT INTO "Employees" VALUES(2,'nancywalker@email.com','+2223334444',1,2,'Nancy Walker');
INSERT INTO "Employees" VALUES(3,'thanks@email.com','+3344556677',2,1,'Tom Hanks');
INSERT INTO "Employees" VALUES(4,'laurawilliams@email.com','+4455667788',2,2,'Laura Williams');
INSERT INTO "Employees" VALUES(5,'stevemartin@email.com','+5566778899',1,1,'Steve Martin');
CREATE TABLE Language_Proficiencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language TEXT
);
INSERT INTO "Language_Proficiencies" VALUES(1,'Bilingual');
INSERT INTO "Language_Proficiencies" VALUES(2,'Multilingual');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Customer_Facing_Roles',2);
INSERT INTO "sqlite_sequence" VALUES('Language_Proficiencies',2);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
COMMIT;
