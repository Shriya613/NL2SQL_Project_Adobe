BEGIN TRANSACTION;
CREATE TABLE ChefRestaurants(
    ChefID INTEGER,
    RestaurantID INTEGER,
    HireDate DATE,
    FOREIGN KEY (ChefID) REFERENCES Chefs(ChefID),
    FOREIGN KEY (RestaurantID) REFERENCES Restaurants(RestaurantID),
    PRIMARY KEY (ChefID, RestaurantID));
INSERT INTO "ChefRestaurants" VALUES(1,1,'2020-01-01');
INSERT INTO "ChefRestaurants" VALUES(2,2,'2021-02-01');
INSERT INTO "ChefRestaurants" VALUES(3,3,'2020-06-01');
INSERT INTO "ChefRestaurants" VALUES(4,4,'2021-08-01');
INSERT INTO "ChefRestaurants" VALUES(5,5,'2020-09-01');
CREATE TABLE Chefs(
    ChefID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    LanguageID INTEGER,
    FOREIGN KEY (LanguageID) REFERENCES Languages(LanguageID));
INSERT INTO "Chefs" VALUES(1,'John','Doe',1);
INSERT INTO "Chefs" VALUES(2,'Mary','Smith',2);
INSERT INTO "Chefs" VALUES(3,'Alice','Brown',3);
INSERT INTO "Chefs" VALUES(4,'Bob','White',4);
INSERT INTO "Chefs" VALUES(5,'Charlie','Black',5);
CREATE TABLE Languages(
    LanguageID INTEGER PRIMARY KEY,
    LanguageName TEXT);
INSERT INTO "Languages" VALUES(1,'English');
INSERT INTO "Languages" VALUES(2,'French');
INSERT INTO "Languages" VALUES(3,'German');
INSERT INTO "Languages" VALUES(4,'Italian');
INSERT INTO "Languages" VALUES(5,'Japanese');
CREATE TABLE Restaurants(
    RestaurantID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT,
    OpeningHours TEXT);
INSERT INTO "Restaurants" VALUES(1,'Le Jardin','123 Main St','555-1234','9am - 10pm');
INSERT INTO "Restaurants" VALUES(2,'La Maison','456 Elm St','555-5678','10am - 11pm');
INSERT INTO "Restaurants" VALUES(3,'Der Biergarten','789 Oak St','555-9012','11am - midnight');
INSERT INTO "Restaurants" VALUES(4,'Il Ristorante','999 Pine St','555-1357','Noon - 11pm');
INSERT INTO "Restaurants" VALUES(5,'The Sushi Bar','321 Maple Ave','555-1415','Lunch & dinner only');
CREATE TABLE ServerRestaurants(
    ServerID INTEGER,
    RestaurantID INTEGER,
    HireDate DATE,
    FOREIGN KEY (ServerID) REFERENCES Servers(ServerID),
    FOREIGN KEY (RestaurantID) REFERENCES Restaurants(RestaurantID),
    PRIMARY KEY (ServerID, RestaurantID));
INSERT INTO "ServerRestaurants" VALUES(1,1,'2020-02-01');
INSERT INTO "ServerRestaurants" VALUES(2,2,'2021-03-01');
INSERT INTO "ServerRestaurants" VALUES(3,3,'2020-07-01');
INSERT INTO "ServerRestaurants" VALUES(4,4,'2021-09-01');
INSERT INTO "ServerRestaurants" VALUES(5,5,'2020-10-01');
CREATE TABLE Servers(
    ServerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    LanguageID INTEGER,
    FOREIGN KEY (LanguageID) REFERENCES Languages(LanguageID));
INSERT INTO "Servers" VALUES(1,'Jack','Green',1);
INSERT INTO "Servers" VALUES(2,'Emma','Blue',2);
INSERT INTO "Servers" VALUES(3,'David','Yellow',3);
INSERT INTO "Servers" VALUES(4,'Lucy','Red',4);
INSERT INTO "Servers" VALUES(5,'Oliver','Orange',5);
COMMIT;
