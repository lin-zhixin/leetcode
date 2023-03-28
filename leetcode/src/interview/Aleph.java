package interview;


import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.stream.Collectors;

public class Aleph {

    /*
    "Order{ long id; String shop; BigDecimal value};
     List<Order> orderList;有下列内容：
     {1,""shopA"",10},{2,""shopB"",10},{3,""shopB"",20},{4,""shopC"",10},{5,""shopC"",20},{6,""shopC"",30}，{7,""shopC"",null}.
             1) 对orderList 按shop分组；
             2) 求各个shop的value之和；
             3) 求各个shop的value的平均值，保留2位小数
             4)求各个shop的value最大、最小值
     考虑真实情况下，可能有的order的属性是null，这种order不作处理，
     尽量使用最简的运算复杂度，循环遍历越少越好"
     */
    public void ordersDeal() {
        List<Order> orderList = new ArrayList<>();
        orderList.add(new Order(1, "shopA", BigDecimal.valueOf(10)));
        orderList.add(new Order(2, "shopB", BigDecimal.valueOf(10)));
        orderList.add(new Order(3, "shopC", BigDecimal.valueOf(20)));
        orderList.add(new Order(4, "shopC", BigDecimal.valueOf(10)));
        orderList.add(new Order(5, "shopC", BigDecimal.valueOf(20)));
        orderList.add(new Order(6, "shopC", BigDecimal.valueOf(30)));
        orderList.add(new Order(7, "shopC", null));
//        1)
        Map<String, List<Order>> map = orderList.stream().collect(Collectors.groupingBy(Order::getShop));
        map.forEach((k, v) -> {
            BigDecimal sum = BigDecimal.valueOf(0);
            BigDecimal count = BigDecimal.valueOf(0);
            BigDecimal avg = BigDecimal.valueOf(0);
            BigDecimal max = new BigDecimal(Double.MIN_VALUE);
            BigDecimal min = new BigDecimal(Double.MAX_VALUE);

            for (Order order : v) {
                BigDecimal val = order.getValue();
                if (Objects.nonNull(val)) {
                    count = count.add(BigDecimal.valueOf(1));
                    sum = sum.add(val);
                    avg = sum.divide(count, RoundingMode.HALF_UP);
                    max = max.compareTo(val) > 0 ? max : val;
                    min = min.compareTo(val) < 0 ? min : val;
                }
            }
//            2)3)4)
            System.out.println(k + "的value之和为：" + sum + " 最大值：" + max + " 最小值：" + min + " 平均值：" + avg);
        });
    }


    /*
        "给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

        说明：

        分隔时可以重复使用字典中的单词。
        你可以假设字典中没有重复的单词。

        示例 1：
        输入:
        s = ""catsanddog""
        wordDict = [""cat"", ""cats"", ""and"", ""sand"", ""dog""]
        输出:，
                [
                ""cats and dog"",
                ""cat sand dog""
                ]

        示例 2：
        输入:
        s = ""pineapplepenapple""
        wordDict = [""apple"", ""pen"", ""applepen"", ""pine"", ""pineapple""]
        输出:
                [
                ""pine apple pen apple"",
                ""pineapple pen apple"",
                ""pine applepen apple""
                ]
        解释: 注意你可以重复使用字典中的单词。

    */
    public void wordBreak(String s, List<String> wordDict) {
        // Map用来防止重复计算 Integer:下标索引 List<String>：s[Integer~]子串所组成的所有结果
        Map<Integer, List<String>> memo = new HashMap<>();
        System.out.println(dp(s, 0, wordDict, memo));
//        return dp(s, 0, wordDict, memo);
    }

    // s[i~]这个串被拼出的所有可能值
    public List<String> dp(String s, int i, List<String> wordDict, Map<Integer, List<String>> memo) {
        List<String> res = new ArrayList<>();
        if (i == s.length()) {
//            必要的 空字符串必须返回一个 "" 否则被认为s[s.length()~]不能被拼出来，之后res永远为空
            res.add("");//保证res有长度 如果什么都不做下面的for (String right : tempRes)没有值
            return res;
        }
        if (memo.containsKey(i)) {
            return memo.get(i);
        }
        // 遍历当前序列
        int len = 1;
        while (len + i <= s.length()) {
            // 找到s[i~]的前缀：s[i~len]，如果s[i~i+len]存在则结果变成s[i~len]+s[i+len~]的所有可能组成
            String left = s.substring(i, i + len);
            if (wordDict.contains(left)) {
//                rights :s[i~i+len]的所有可能组合，只要加上当前的前缀就是s[i~]的所有可能组合
                List<String> rights = dp(s, i + len, wordDict, memo);
                rights.forEach(right -> res.add(right.isEmpty() ? left : left + " " + right));
                memo.put(i, res);
            }
            len++;
        }
        return res;
    }

    public static void main(String[] args) {
        Aleph alep = new Aleph();
        alep.ordersDeal();
        System.out.println();
        List<String> wordDict = new ArrayList<>();
        wordDict.add("apple");
        wordDict.add("pen");
        wordDict.add("applepen");
        wordDict.add("pine");
        wordDict.add("pineapple");
        alep.wordBreak("pineapplepenapple", wordDict);
    }

}

class Order {
    long id;
    String shop;
    BigDecimal value;

    Order(long id, String shop, BigDecimal value) {
        this.id = id;
        this.shop = shop;
        this.value = value;
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
