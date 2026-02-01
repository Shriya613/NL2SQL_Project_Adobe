BEGIN TRANSACTION;
CREATE TABLE ATM_Machine(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location_Within_Terminal TEXT,
    Operating_Company TEXT,
    Fees REAL,
    Language_Options TEXT);
INSERT INTO "ATM_Machine" VALUES(1,'ATM Near Gate 10','Near Gate 10','ABCD Bank',3.0,'English, Spanish, French');
INSERT INTO "ATM_Machine" VALUES(2,'Airport ATM','Departure Level','EFG Bank',2.5,'English, German, Chinese');
INSERT INTO "ATM_Machine" VALUES(3,'Express Cash','Before Security','HIJ Bank',3.25,'English, Russian, Japanese');
INSERT INTO "ATM_Machine" VALUES(4,'Quick Cash','Arrival Area','KLM Bank',3.0,'English, Dutch, Arabic');
INSERT INTO "ATM_Machine" VALUES(5,'Easy Pay','Between Concourse A & B','OPQ Bank',3.5,'English, Italian, Portuguese');
CREATE TABLE Baggage_Claim(
    id INTEGER PRIMARY KEY,
    Terminal TEXT,
    Carousel_Numbers TEXT,
    Flight_Info TEXT,
    Lost_And_Found TEXT);
INSERT INTO "Baggage_Claim" VALUES(1,'Terminal 1','Carousels 1-5','Flight 123','Lost And Found Counter near Carousel 1');
INSERT INTO "Baggage_Claim" VALUES(2,'Terminal 2','Carousels 6-10','Flight 456','Lost And Found Counter near Carousel 6');
INSERT INTO "Baggage_Claim" VALUES(3,'Terminal 3','Carousels 11-15','Flight 789','Lost And Found Counter near Carousel 11');
INSERT INTO "Baggage_Claim" VALUES(4,'Terminal 4','Carousels 16-20','Flight 001','Lost And Found Counter near Carousel 16');
INSERT INTO "Baggage_Claim" VALUES(5,'Terminal 5','Carousels 21-25','Flight 999','Lost And Found Counter near Carousel 21');
CREATE TABLE Currency_Exchange_Service(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location_Within_Terminal TEXT,
    Hours_of_Operation TEXT,
    Phone_Number TEXT,
    Email TEXT);
INSERT INTO "Currency_Exchange_Service" VALUES(1,'Global Exchange','Near Gate 10','8am - 6pm','+123456789','globalexchange@example.com');
INSERT INTO "Currency_Exchange_Service" VALUES(2,'Airport Foreign Exchange','Main Hall','9am - 8pm','+123456789','airporexchange@example.com');
INSERT INTO "Currency_Exchange_Service" VALUES(3,'Travelex','International Departures','10am - 6pm','+123456789','travelex@example.com');
INSERT INTO "Currency_Exchange_Service" VALUES(4,'Thomas Cook','Domestic Arrivals','7am - 9pm','+123456789','thomascook@example.com');
INSERT INTO "Currency_Exchange_Service" VALUES(5,'Cambio','Gate C','24 hours','+123456789','cambio@example.com');
CREATE TABLE Food_and_Beverage(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Type TEXT,
    Location_Within_Terminal TEXT,
    Menu TEXT,
    Contact_Information TEXT);
INSERT INTO "Food_and_Beverage" VALUES(1,'Starbucks','Coffee Shop','Gate D','Espresso, Frappuccino, Sandwiches, Pastries','starbucks@example.com');
INSERT INTO "Food_and_Beverage" VALUES(2,'McDonalds','Fast Food','Concourse E','Burgers, Fries, Chicken Nuggets, Salads','mcdonalds@example.com');
INSERT INTO "Food_and_Beverage" VALUES(3,'Subway','Sandwich Bar','Pre-Security','Subs, Wraps, Salads, Cookies','subway@example.com');
INSERT INTO "Food_and_Beverage" VALUES(4,'Pizza Place','Italian Restaurant','After Security','Pizzas, Pasta, Antipasti, Tiramisu','pizzaplace@example.com');
INSERT INTO "Food_and_Beverage" VALUES(5,'Grill House','BBQ Restaurant','Gate F','Ribs, Steaks, Burgers, BBQ Sauce','grillhouse@example.com');
CREATE TABLE Parking(
    id INTEGER PRIMARY KEY,
    Lot_Name TEXT,
    Location TEXT,
    Daily_Rate REAL,
    Weekly_Rate REAL,
    Monthly_Rate REAL);
INSERT INTO "Parking" VALUES(1,'Short Stay','Next to terminal building',10.0,50.0,250.0);
INSERT INTO "Parking" VALUES(2,'Long Stay','Across the street',8.0,40.0,200.0);
INSERT INTO "Parking" VALUES(3,'Express Park','Outside airport perimeter',5.0,25.0,125.0);
INSERT INTO "Parking" VALUES(4,'Valet Parking','In front of main entrance',20.0,100.0,400.0);
INSERT INTO "Parking" VALUES(5,'Overnight Parking','Designated area',15.0,75.0,350.0);
CREATE TABLE Shops(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Category TEXT,
    Location_Within_Terminal TEXT,
    Product_Offerings TEXT,
    Business_Hours TEXT);
INSERT INTO "Shops" VALUES(1,'Duty Free Store','Luxury Goods','Departure Lounge','Perfumes, Watches, Alcohol, Tobacco','8am - 8pm');
INSERT INTO "Shops" VALUES(2,'Bookshop','Retail','Before Security','Magazines, Novels, Travel Guides','6am - 10pm');
INSERT INTO "Shops" VALUES(3,'Pharmacy','Health Care','Main Concourse','Medications, Toiletries, Cosmetics','7am - 9pm');
INSERT INTO "Shops" VALUES(4,'Clothing Store','Fashion','Between Gates A & B','Tshirts, Jeans, Accessories','8am - 8pm');
INSERT INTO "Shops" VALUES(5,'Electronics Shop','Technology','Gate C','Smartphones, Tablets, Headphones','8am - 8pm');
COMMIT;
