package category;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

public class Stacks {
//    946. 验证栈序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
//        可优化 不用这么多队列
        Stack<Integer> pu=new Stack<Integer>();
        List<Integer> c=Arrays.stream(pushed).boxed().collect(Collectors.toList());
        Collections.reverse(c);
        pu.addAll(c);
        Stack<Integer> po=new Stack<Integer>();
        List<Integer> c1=Arrays.stream(popped).boxed().collect(Collectors.toList());
        Collections.reverse(c1);
        po.addAll(c1);
        Stack<Integer> t=new Stack<Integer>();

        while (!pu.empty()){
            t.push(pu.pop());
            System.out.println(po.peek()+" t:"+t.peek());
            while ((!t.empty())&&(!po.empty())&&t.peek().intValue()==po.peek().intValue()){
                t.pop();
                po.pop();
            }
        }
        return t.empty();
    }

//    "Order{ long id; String shop; BigDecimal value};
//    List<Order> orderList;有下列内容：
//    {1,""shopA"",10},{2,""shopB"",10},{3,""shopB"",20},{4,""shopC"",10},{5,""shopC"",20},{6,""shopC"",30}，{7,""shopC"",null}.
//            1) 对orderList 按shop分组；
//            2) 求各个shop的value之和；
//            3) 求各个shop的value的平均值，保留2位小数
//4)求各个shop的value最大、最小值
//    考虑真实情况下，可能有的order的属性是null，这种order不作处理，
//    尽量使用最简的运算复杂度，循环遍历越少越好"
    public static void main(String[] args) {
        List<Order> orderList=new ArrayList<>();
        orderList.add(new Order(1,"shopA",BigDecimal.valueOf(10)));
        orderList.add(new Order(2,"shopB",BigDecimal.valueOf(10)));
        orderList.add(new Order(3,"shopC",BigDecimal.valueOf(20)));
        orderList.add(new Order(4,"shopC",BigDecimal.valueOf(10)));
        orderList.add(new Order(5,"shopC",BigDecimal.valueOf(20)));
        orderList.add(new Order(6,"shopC",BigDecimal.valueOf(30)));
        orderList.add(new Order(7,"shopC",null));
//        1)
        Map<String, List<Order>> map=orderList.stream().collect(Collectors.groupingBy(Order::getShop));
        map.forEach((k,v)->{
            BigDecimal sum=BigDecimal.valueOf(0);
            BigDecimal count= BigDecimal.valueOf(0);
            BigDecimal avg= BigDecimal.valueOf(0);
            BigDecimal max=new BigDecimal(Double.MIN_VALUE);
            BigDecimal min=new BigDecimal(Double.MAX_VALUE);

            for (Order order : v) {
                BigDecimal val = order.getValue();
                if (Objects.nonNull(val)) {
                    count=count.add(BigDecimal.valueOf(1));
                    sum=sum.add(val);
                    avg=sum.divide(count,BigDecimal.ROUND_HALF_UP);
                    max = max.compareTo(val) > 0 ? max : val;
                    min = min.compareTo(val) < 0 ? min : val;
                }
            }

            System.out.println(k+"的value之和为："+sum+" 最大值："+max+" 最小值："+min+" 平均值："+avg);
        });
    }




//    "给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。
//
//    说明：
//
//    分隔时可以重复使用字典中的单词。
//    你可以假设字典中没有重复的单词。
//
//    示例 1：
//    输入:
//    s = ""catsanddog""
//    wordDict = [""cat"", ""cats"", ""and"", ""sand"", ""dog""]
//    输出:，
//            [
//            ""cats and dog"",
//            ""cat sand dog""
//            ]
//
//    示例 2：
//    输入:
//    s = ""pineapplepenapple""
//    wordDict = [""apple"", ""pen"", ""applepen"", ""pine"", ""pineapple""]
//    输出:
//            [
//            ""pine apple pen apple"",
//            ""pineapple pen apple"",
//            ""pine applepen apple""
//            ]
//    解释: 注意你可以重复使用字典中的单词。
}
