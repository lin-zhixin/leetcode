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

}
