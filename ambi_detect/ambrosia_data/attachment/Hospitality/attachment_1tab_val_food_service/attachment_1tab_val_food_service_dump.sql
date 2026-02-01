BEGIN TRANSACTION;
CREATE TABLE Orders(
    ID INTEGER PRIMARY KEY,
    OrderDate DATETIME,
    Item TEXT,
    Quantity INTEGER,
    Price REAL,
    ReservationID INTEGER,
    FOREIGN KEY(ReservationID) REFERENCES Reservations(ID));
INSERT INTO "Orders" VALUES(1,'2023-04-01 18:30:00','Margherita Pizza',2,10.0,1);
INSERT INTO "Orders" VALUES(2,'2023-04-02 19:00:00','Spaghetti Carbonara',4,15.0,2);
INSERT INTO "Orders" VALUES(3,'2023-04-03 20:00:00','Chicken Caesar Salad',6,12.0,3);
INSERT INTO "Orders" VALUES(4,'2023-04-04 17:00:00','Garlic Bread',3,5.0,4);
INSERT INTO "Orders" VALUES(5,'2023-04-05 19:00:00','Tiramisu',8,8.0,5);
CREATE TABLE Reservations(
    ID INTEGER PRIMARY KEY,
    Date DATETIME,
    Time TIME,
    TableNumber INTEGER,
    Guests INTEGER,
    TotalCost REAL,
    CustomerName TEXT,
    TableID INTEGER,
    FOREIGN KEY(TableID) REFERENCES Tables(ID));
INSERT INTO "Reservations" VALUES(1,'2023-04-01 18:30:00','19:00:00',4,2,120.0,'John Smith',1);
INSERT INTO "Reservations" VALUES(2,'2023-04-02 19:00:00','20:00:00',6,4,240.0,'Jane Doe',2);
INSERT INTO "Reservations" VALUES(3,'2023-04-03 20:00:00','21:00:00',8,6,300.0,'Alice Johnson',3);
INSERT INTO "Reservations" VALUES(4,'2023-04-04 17:00:00','18:00:00',10,3,90.0,'Bob Brown',4);
INSERT INTO "Reservations" VALUES(5,'2023-04-05 19:00:00','20:00:00',12,8,320.0,'Emma Wilson',5);
CREATE TABLE Restaurants(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    Food_Service TEXT,
    Cuisine TEXT);
INSERT INTO "Restaurants" VALUES(1,'Ristorante Pizzeria','Viale XX Settembre 27','Restaurant','Italian');
INSERT INTO "Restaurants" VALUES(2,'La Casa Bianca','Corso Umberto I 80','Restaurant','Mexican');
INSERT INTO "Restaurants" VALUES(3,'Le Café Français','Piazza Duomo 6','Café','French');
INSERT INTO "Restaurants" VALUES(4,'Il Giardino Romano','Lungotevere delle Armi 9','Café','Italian');
INSERT INTO "Restaurants" VALUES(5,'The English Tea Room','King Street 5','Café','English');
CREATE TABLE Tables(
    ID INTEGER PRIMARY KEY,
    Number INTEGER,
    Status TEXT ,
    RestaurantID INTEGER,
    FOREIGN KEY(RestaurantID) REFERENCES Restaurants(ID));
INSERT INTO "Tables" VALUES(1,4,'Available',1);
INSERT INTO "Tables" VALUES(2,6,'Occupied',2);
INSERT INTO "Tables" VALUES(3,8,'Available',3);
INSERT INTO "Tables" VALUES(4,10,'Under Maintenance',4);
INSERT INTO "Tables" VALUES(5,12,'Unavailable',5);
COMMIT;
