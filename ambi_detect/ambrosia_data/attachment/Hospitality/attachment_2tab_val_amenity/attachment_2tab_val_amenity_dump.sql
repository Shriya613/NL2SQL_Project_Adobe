BEGIN TRANSACTION;
CREATE TABLE Accommodations (
    id INTEGER PRIMARY KEY,
    Type TEXT, -- Options can include Hotel, Hostel, Bed & Breakfast, Vacation Home, Resort, etc.
    Capacity INTEGER ,
    Amenities TEXT, -- Can be a comma-separated list of amenities like Pool, WiFi, Parking, etc.
    Beach_id INTEGER,
    FOREIGN KEY(Beach_id) REFERENCES Beaches(id) ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "Accommodations" VALUES(1,'Hotel',50,'Pool, WiFi, Parking',1);
INSERT INTO "Accommodations" VALUES(2,'Vacation Home',8,'No Pets Allowed, Jacuzzi, BBQ',2);
CREATE TABLE Beach_Clubs (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    OpeningHours TEXT,
    PriceRange TEXT,
    Beach_id INTEGER,
    FOREIGN KEY(Beach_id) REFERENCES Beaches(id) ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "Beach_Clubs" VALUES(1,'Paradise Club','White Sands Beach','9am - 6pm','$$',1);
INSERT INTO "Beach_Clubs" VALUES(2,'Ocean View Club','Sunset Shores','10am - 7pm','$$$',2);
CREATE TABLE Beaches (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Beaches" VALUES(1,'White Sands Beach','123 Main St','555-555-5555','whitesand@email.com');
INSERT INTO "Beaches" VALUES(2,'Sunset Shores','456 Ocean Blvd','555-555-6789','sunsets@email.com');
CREATE TABLE Hotels (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Star_rating INTEGER ,
    PriceRange TEXT,
    Beach_id INTEGER,
    FOREIGN KEY(Beach_id) REFERENCES Beaches(id) ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "Hotels" VALUES(1,'Oceanfront Inn',3,'Moderate',1);
INSERT INTO "Hotels" VALUES(2,'Shoreline Suites',4,'High',2);
CREATE TABLE Restaurants (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Cuisine TEXT,
    Rating REAL ,
    Beach_id INTEGER,
    FOREIGN KEY(Beach_id) REFERENCES Beaches(id) ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "Restaurants" VALUES(1,'Seaside Grill','Italian',4.5,1);
INSERT INTO "Restaurants" VALUES(2,'Surfside Cafe','Mexican',4.0,2);
CREATE TABLE "Watersports_Rentals" (
    id INTEGER PRIMARY KEY, `name` TEXT,
    Item TEXT,
    RentalRate REAL,
    AvailableDays TEXT,
    Beach_id INTEGER,
    FOREIGN KEY(Beach_id) REFERENCES Beaches(id) ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "Watersports_Rentals" VALUES(1,'AquaAdventures Rentals','Kayak',20.0,'Mon - Fri',1);
INSERT INTO "Watersports_Rentals" VALUES(2,'WaveRider Rentals','Paddleboard',25.0,'All Days',2);
COMMIT;
