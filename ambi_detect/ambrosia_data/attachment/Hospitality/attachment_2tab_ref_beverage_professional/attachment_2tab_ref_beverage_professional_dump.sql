BEGIN TRANSACTION;
CREATE TABLE Bartenders(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurantId INTEGER, FullName TEXT,
    FOREIGN KEY(restaurantId) REFERENCES Restaurants(id));
INSERT INTO "Bartenders" VALUES(1,1,'Alex Martin');
INSERT INTO "Bartenders" VALUES(2,1,'David Lee');
INSERT INTO "Bartenders" VALUES(3,2,'Amanda Green');
INSERT INTO "Bartenders" VALUES(4,2,'Chris Wright');
INSERT INTO "Bartenders" VALUES(5,3,'Karen Hall');
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    phoneNumber TEXT);
INSERT INTO "Customers" VALUES(1,'Tom','Brown','555-0001');
INSERT INTO "Customers" VALUES(2,'Lucy','White','555-0002');
INSERT INTO "Customers" VALUES(3,'Jack','Black','555-0003');
INSERT INTO "Customers" VALUES(4,'Anna','Blue','555-0004');
INSERT INTO "Customers" VALUES(5,'Samantha','Red','555-0005');
CREATE TABLE Restaurants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    phoneNumber TEXT);
INSERT INTO "Restaurants" VALUES(1,'Le Jardin','123 Main St','555-1234');
INSERT INTO "Restaurants" VALUES(2,'The Rooftop Lounge','456 Elm St','555-5678');
INSERT INTO "Restaurants" VALUES(3,'Café des Amis','789 Oak St','555-9012');
INSERT INTO "Restaurants" VALUES(4,'La Maison du Fromage','135 Pine St','555-1357');
INSERT INTO "Restaurants" VALUES(5,'Gourmet Bistro','321 Maple Ave','555-2345');
CREATE TABLE Sommeliers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurantId INTEGER, FullName TEXT,
    FOREIGN KEY(restaurantId) REFERENCES Restaurants(id));
INSERT INTO "Sommeliers" VALUES(1,1,'Alex Dupont');
INSERT INTO "Sommeliers" VALUES(2,1,'Michael Johnson');
INSERT INTO "Sommeliers" VALUES(3,2,'Nina Smith');
INSERT INTO "Sommeliers" VALUES(4,2,'Peter Jones');
INSERT INTO "Sommeliers" VALUES(5,3,'Emily Williams');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Restaurants',5);
INSERT INTO "sqlite_sequence" VALUES('Sommeliers',5);
INSERT INTO "sqlite_sequence" VALUES('Bartenders',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
COMMIT;
