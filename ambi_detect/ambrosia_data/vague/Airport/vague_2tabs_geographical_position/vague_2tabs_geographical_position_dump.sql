BEGIN TRANSACTION;
CREATE TABLE Airline (
    id INTEGER PRIMARY KEY,
    name TEXT,
    headquarter_city_id INTEGER,
    FOREIGN KEY(headquarter_city_id) REFERENCES City(id)
);
INSERT INTO "Airline" VALUES(1,'American Airlines',1);
INSERT INTO "Airline" VALUES(2,'British Airways',2);
INSERT INTO "Airline" VALUES(3,'All Nippon Airways',3);
INSERT INTO "Airline" VALUES(4,'Air France-KLM',4);
INSERT INTO "Airline" VALUES(5,'Emirates',5);
CREATE TABLE Airport (
    id INTEGER PRIMARY KEY,
    name TEXT,
    city_id INTEGER,
    coordinates_id INTEGER,
    FOREIGN KEY(city_id) REFERENCES City(id),
    FOREIGN KEY(coordinates_id) REFERENCES Coordinates(id)
);
INSERT INTO "Airport" VALUES(1,'John F. Kennedy International Airport',1,1);
INSERT INTO "Airport" VALUES(2,'Heathrow Airport',2,2);
INSERT INTO "Airport" VALUES(3,'Narita International Airport',3,3);
INSERT INTO "Airport" VALUES(4,'Charles de Gaulle Airport',4,4);
INSERT INTO "Airport" VALUES(5,'Dubai International Airport',5,5);
CREATE TABLE City (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT
);
INSERT INTO "City" VALUES(1,'New York','USA');
INSERT INTO "City" VALUES(2,'London','UK');
INSERT INTO "City" VALUES(3,'Tokyo','Japan');
INSERT INTO "City" VALUES(4,'Paris','France');
INSERT INTO "City" VALUES(5,'Dubai','UAE');
CREATE TABLE Coordinates (
    id INTEGER PRIMARY KEY,
    latitude REAL,
    longitude REAL
);
INSERT INTO "Coordinates" VALUES(1,-73.985428,-109.136635);
INSERT INTO "Coordinates" VALUES(2,51.507223,-0.127503);
INSERT INTO "Coordinates" VALUES(3,35.689501,139.691699);
INSERT INTO "Coordinates" VALUES(4,48.8566,2.352222);
INSERT INTO "Coordinates" VALUES(5,25.27439,55.29888);
CREATE TABLE Flight (
    id INTEGER PRIMARY KEY,
    airport_from_id INTEGER,
    airport_to_id INTEGER,
    airline_id INTEGER,
    departure_time DATETIME,
    arrival_time DATETIME,
    FOREIGN KEY(airport_from_id) REFERENCES Airport(id),
    FOREIGN KEY(airport_to_id) REFERENCES Airport(id),
    FOREIGN KEY(airline_id) REFERENCES Airline(id)
);
INSERT INTO "Flight" VALUES(1,1,2,1,'2022-01-01 12:00:00','2022-01-01 18:00:00');
INSERT INTO "Flight" VALUES(2,3,4,2,'2022-01-01 08:00:00','2022-01-01 14:00:00');
INSERT INTO "Flight" VALUES(3,5,1,3,'2022-01-02 12:00:00','2022-01-02 18:00:00');
INSERT INTO "Flight" VALUES(4,2,3,4,'2022-01-02 08:00:00','2022-01-02 14:00:00');
INSERT INTO "Flight" VALUES(5,4,5,5,'2022-01-03 12:00:00','2022-01-03 18:00:00');
CREATE TABLE Runway (
    id INTEGER PRIMARY KEY,
    airport_id INTEGER,
    length INTEGER,
    width INTEGER,
    surface_type TEXT,
    FOREIGN KEY(airport_id) REFERENCES Airport(id)
);
INSERT INTO "Runway" VALUES(1,1,10000,200,'Asphalt');
INSERT INTO "Runway" VALUES(2,2,10000,200,'Concrete');
INSERT INTO "Runway" VALUES(3,3,9000,200,'Asphalt');
INSERT INTO "Runway" VALUES(4,4,9000,200,'Concrete');
INSERT INTO "Runway" VALUES(5,5,12000,200,'Asphalt');
COMMIT;
