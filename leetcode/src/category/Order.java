package category;

import java.math.BigDecimal;

class Order{
    long id;
    String shop;
    BigDecimal value;
    Order(long id, String shop,BigDecimal value){
        this.id=id;
        this.shop=shop;
        this.value=value;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getShop() {
        return shop;
    }

    public void setShop(String shop) {
        this.shop = shop;
    }

    public BigDecimal getValue() {
        return value;
    }

    public void setValue(BigDecimal value) {
        this.value = value;
    }
}