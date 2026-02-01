BEGIN TRANSACTION;
CREATE TABLE MenuItems(
    ItemID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Category TEXT,
    Price DECIMAL(10,2),
    Description TEXT,
    Restaurant_ID INTEGER,
    FOREIGN KEY (Restaurant_ID) REFERENCES Restaurant(RestID)
);
INSERT INTO "MenuItems" VALUES(1,'Margherita Pizza','Pizza',12.99,'Classic Margherita with mozzarella, tomato sauce, basil.',1);
INSERT INTO "MenuItems" VALUES(2,'Enchiladas Suizas','Mexican',10.99,'Three corn tortillas filled with chicken and smothered in a creamy green salsa verde, garnished with sour cream and onions.',2);
INSERT INTO "MenuItems" VALUES(3,'Kung Pao Chicken','Chinese',13.99,'Stir-fried diced chicken breast with peanuts, dried red chili pepper, and vegetables in a spicy Kung Pao sauce.',3);
INSERT INTO "MenuItems" VALUES(4,'Biryani','Indian',11.99,'Aromatic rice dish made with long grain basmati rice, layered with spices and cooked with meat.',4);
INSERT INTO "MenuItems" VALUES(5,'Cheeseburger','American',8.99,'Grilled beef patty served on a sesame seed bun with lettuce, tomatoes, onions, pickles, cheese, and mayonnaise.',5);
CREATE TABLE Reservations(
    ReservationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Date DATETIME,
    Time TIME,
    No_Of_Guests INT,
    Customer_Name TEXT,
    TableID INTEGER,
    FOREIGN KEY (TableID) REFERENCES Seating_Areas(AreaID)
);
INSERT INTO "Reservations" VALUES(1,'2022-01-01 18:30','5 PM',4,'John Doe',1);
INSERT INTO "Reservations" VALUES(2,'2022-01-02 20:00','8 PM',2,'Jane Smith',2);
INSERT INTO "Reservations" VALUES(3,'2022-01-03 19:30','7:30 PM',6,'Emily Johnson',3);
INSERT INTO "Reservations" VALUES(4,'2022-01-04 18:00','6 PM',8,'William Brown',4);
INSERT INTO "Reservations" VALUES(5,'2022-01-05 20:30','9 PM',10,'Olivia Wilson',5);
CREATE TABLE Restaurant(
    RestID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    Contact_No TEXT,
    Seating_Area_ID INTEGER,
    FOREIGN KEY (Seating_Area_ID) REFERENCES Seating_Areas(AreaID)
);
INSERT INTO "Restaurant" VALUES(1,'Ristorante Italiano','123 Main St','+1 800 1234567',1);
INSERT INTO "Restaurant" VALUES(2,'Taco Town','456 Elm St','+1 987 6543210',2);
INSERT INTO "Restaurant" VALUES(3,'Wok This Way','789 Oak St','+1 234 5678901',3);
INSERT INTO "Restaurant" VALUES(4,'Samosa Palace','135 Pine St','+1 345 6789012',4);
INSERT INTO "Restaurant" VALUES(5,'All American Grill','987 Birch St','+1 456 7890123',5);
CREATE TABLE Reviews(
    ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    Rating INT ,
    Comment TEXT,
    Reservation_ID INTEGER,
    FOREIGN KEY (Reservation_ID) REFERENCES Reservations(ReservationID)
);
INSERT INTO "Reviews" VALUES(1,5,'Great food, amazing service!',1);
INSERT INTO "Reviews" VALUES(2,4,'The ambiance was lovely but I found my meal a bit overpriced.',2);
INSERT INTO "Reviews" VALUES(3,3,'Nice place to hang out but could use some improvement on their menu.',3);
INSERT INTO "Reviews" VALUES(4,5,'We had an excellent time at this restaurant. The staff were very friendly and attentive.',4);
INSERT INTO "Reviews" VALUES(5,4,'Really enjoyed our visit here. We will definitely come back again.',5);
CREATE TABLE Seating_Areas(
    AreaID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Number_Of_Seats INT,
    Type_of_Food TEXT
);
INSERT INTO "Seating_Areas" VALUES(1,'Patio',50,'Italian');
INSERT INTO "Seating_Areas" VALUES(2,'Patio',60,'Mexican');
INSERT INTO "Seating_Areas" VALUES(3,'Veranda',50,'Chinese');
INSERT INTO "Seating_Areas" VALUES(4,'Veranda',30,'Indian');
INSERT INTO "Seating_Areas" VALUES(5,'Lobby',20,'American');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Seating_Areas',5);
INSERT INTO "sqlite_sequence" VALUES('Restaurant',5);
INSERT INTO "sqlite_sequence" VALUES('MenuItems',5);
INSERT INTO "sqlite_sequence" VALUES('Reservations',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
COMMIT;
